package agent

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
)

// CLIAgent invokes a local CLI agent (claude, codex, etc.) via streaming JSON.
type CLIAgent struct {
	name         string
	command      string
	args         []string          // extra args from config
	cwd          string            // working directory
	env          map[string]string // extra environment variables
	model        string
	systemPrompt string
	mu           sync.Mutex
	sessions     map[string]string // conversationID -> session ID for multi-turn
}

// CLIAgentConfig holds configuration for a CLI agent.
type CLIAgentConfig struct {
	Name         string            // agent name for logging, e.g. "claude", "codex"
	Command      string            // path to binary
	Args         []string          // extra args (e.g. ["--dangerously-skip-permissions"])
	Cwd          string            // working directory (workspace)
	Env          map[string]string // extra environment variables
	Model        string
	SystemPrompt string
}

// NewCLIAgent creates a new CLI agent.
func NewCLIAgent(cfg CLIAgentConfig) *CLIAgent {
	cwd := cfg.Cwd
	if cwd == "" {
		cwd = defaultWorkspace()
	}
	return &CLIAgent{
		name:         cfg.Name,
		command:      cfg.Command,
		args:         cfg.Args,
		cwd:          cwd,
		env:          cfg.Env,
		model:        cfg.Model,
		systemPrompt: cfg.SystemPrompt,
		sessions:     make(map[string]string),
	}
}

// streamEvent represents a single event from claude's stream-json output.
type streamEvent struct {
	Type      string         `json:"type"`
	SessionID string         `json:"session_id"`
	Result    string         `json:"result"`
	IsError   bool           `json:"is_error"`
	Message   *streamMessage `json:"message,omitempty"`
}

// streamMessage represents the message field in an assistant event.
type streamMessage struct {
	Content []streamContent `json:"content"`
}

// streamContent represents a content block in an assistant message.
type streamContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// Info returns metadata about this agent.
func (a *CLIAgent) Info() AgentInfo {
	return AgentInfo{
		Name:    a.name,
		Type:    "cli",
		Model:   a.model,
		Command: a.command,
	}
}

// ResetSession clears the existing session for the given conversationID.
// Returns an empty string because the new session ID is only known after the
// next Chat call (claude assigns it during the conversation).
func (a *CLIAgent) ResetSession(_ context.Context, conversationID string) (string, error) {
	a.mu.Lock()
	delete(a.sessions, conversationID)
	a.mu.Unlock()
	if err := os.Remove(a.sessionPath(conversationID)); err != nil && !os.IsNotExist(err) {
		log.Printf("[cli] failed to remove session file (command=%s, conversation=%s): %v", a.command, conversationID, err)
	}
	log.Printf("[cli] session reset (command=%s, conversation=%s)", a.command, conversationID)
	return "", nil
}

// SetCwd changes the working directory for subsequent CLI invocations.
func (a *CLIAgent) SetCwd(cwd string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.cwd = cwd
}

// Chat sends a message to the CLI agent and returns the response.
func (a *CLIAgent) Chat(ctx context.Context, conversationID string, message string) (string, error) {
	switch a.name {
	case "codex":
		return a.chatCodex(ctx, conversationID, message)
	default:
		return a.chatClaude(ctx, conversationID, message)
	}
}

// chatClaude uses claude -p with stream-json to get structured output and session persistence.
func (a *CLIAgent) chatClaude(ctx context.Context, conversationID string, message string) (string, error) {
	args := []string{"-p", message, "--output-format", "stream-json", "--verbose"}

	if a.model != "" {
		args = append(args, "--model", a.model)
	}
	if a.systemPrompt != "" {
		args = append(args, "--append-system-prompt", a.systemPrompt)
	}
	// Append extra args from config (e.g. --dangerously-skip-permissions)
	args = append(args, a.args...)

	// Resume existing session for multi-turn conversation
	a.mu.Lock()
	sessionID, hasSession := a.sessions[conversationID]
	a.mu.Unlock()
	if !hasSession {
		if loadedSessionID, ok := a.loadSession(conversationID); ok {
			sessionID = loadedSessionID
			hasSession = true
			a.mu.Lock()
			a.sessions[conversationID] = sessionID
			a.mu.Unlock()
		} else {
			log.Printf("[cli] no persisted session found (agent=%s, conversation=%s, path=%s)", a.name, conversationID, a.sessionPath(conversationID))
		}
	}

	if hasSession {
		args = append(args, "--resume", sessionID)
		log.Printf("[cli] resuming session (command=%s, session=%s, conversation=%s)", a.command, sessionID, conversationID)
	} else {
		log.Printf("[cli] starting new conversation (command=%s, conversation=%s)", a.command, conversationID)
	}

	cmd := exec.CommandContext(ctx, a.command, args...)
	if a.cwd != "" {
		cmd.Dir = a.cwd
	}
	if len(a.env) > 0 {
		cmdEnv, err := mergeEnv(os.Environ(), a.env)
		if err != nil {
			return "", fmt.Errorf("build %s env: %w", a.name, err)
		}
		cmd.Env = cmdEnv
	}
	var stderr strings.Builder
	cmd.Stderr = &stderr

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return "", fmt.Errorf("create stdout pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		return "", fmt.Errorf("start %s: %w", a.name, err)
	}

	log.Printf("[cli] spawned process (command=%s, pid=%d, conversation=%s)", a.command, cmd.Process.Pid, conversationID)

	// Parse streaming JSON events
	var result string
	var newSessionID string
	var assistantTexts []string

	scanner := bufio.NewScanner(stdout)
	scanner.Buffer(make([]byte, 0, 1024*1024), 1024*1024) // 1MB buffer for large responses

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}

		var event streamEvent
		if err := json.Unmarshal([]byte(line), &event); err != nil {
			continue
		}

		// Capture session ID from any event
		if event.SessionID != "" {
			newSessionID = event.SessionID
		}

		switch event.Type {
		case "result":
			if event.IsError {
				return "", fmt.Errorf("%s returned error: %s", a.name, event.Result)
			}
			result = event.Result
		case "assistant":
			// Newer claude CLI versions send text in assistant events
			// instead of the result event's result field.
			if event.Message != nil {
				for _, c := range event.Message.Content {
					if c.Type == "text" && c.Text != "" {
						assistantTexts = append(assistantTexts, c.Text)
					}
				}
			}
		}
	}

	// If the result event had an empty result, fall back to accumulated assistant texts.
	if result == "" && len(assistantTexts) > 0 {
		result = strings.Join(assistantTexts, "")
	}

	if err := cmd.Wait(); err != nil {
		if result == "" {
			errMsg := strings.TrimSpace(stderr.String())
			if errMsg != "" {
				return "", fmt.Errorf("%s exited with error: %w, stderr: %s", a.name, err, errMsg)
			}
			return "", fmt.Errorf("%s exited with error: %w", a.name, err)
		}
		// If we got a result but exit code is non-zero (e.g. hook failures), still return the result
	}

	log.Printf("[cli] process exited (command=%s, pid=%d)", a.command, cmd.Process.Pid)

	// Save session ID for multi-turn conversation
	if newSessionID != "" {
		a.mu.Lock()
		a.sessions[conversationID] = newSessionID
		a.mu.Unlock()
		log.Printf("[cli] saved session (session=%s, conversation=%s)", newSessionID, conversationID)
	}

	result = strings.TrimSpace(result)
	if result == "" {
		return "", fmt.Errorf("%s returned empty response", a.name)
	}

	return result, nil
}

type codexJSONEvent struct {
	Type     string `json:"type"`
	ThreadID string `json:"thread_id"`
	Item     *struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"item,omitempty"`
}

// chatCodex handles codex CLI invocation using "codex exec" and persists
// thread IDs per WeChat conversation for multi-turn context.
func (a *CLIAgent) chatCodex(ctx context.Context, conversationID string, message string) (string, error) {
	outFile := filepath.Join(os.TempDir(), fmt.Sprintf("weclaw-codex-%d.txt", os.Getpid()))
	_ = os.Remove(outFile)
	defer os.Remove(outFile)

	a.mu.Lock()
	sessionID, hasSession := a.sessions[conversationID]
	a.mu.Unlock()
	if !hasSession {
		if loadedSessionID, ok := a.loadSession(conversationID); ok {
			sessionID = loadedSessionID
			hasSession = true
			a.mu.Lock()
			a.sessions[conversationID] = sessionID
			a.mu.Unlock()
		} else {
			log.Printf("[cli] no persisted codex thread found (agent=%s, conversation=%s, path=%s)", a.name, conversationID, a.sessionPath(conversationID))
		}
	}

	var args []string
	if hasSession {
		args = []string{"exec", "resume", "--json", "--output-last-message", outFile}
		if a.model != "" {
			args = append(args, "--model", a.model)
		}
		args = append(args, filterCodexResumeArgs(a.args)...)
		args = append(args, sessionID, message)
		log.Printf("[cli] resuming codex thread (command=%s, thread=%s, conversation=%s)", a.command, sessionID, conversationID)
	} else {
		args = []string{"exec", "--json", "--output-last-message", outFile}
		if a.model != "" {
			args = append(args, "--model", a.model)
		}
		args = append(args, a.args...)
		args = append(args, message)
		log.Printf("[cli] starting new codex thread (command=%s, conversation=%s)", a.command, conversationID)
	}

	cmd := exec.CommandContext(ctx, a.command, args...)
	if a.cwd != "" {
		cmd.Dir = a.cwd
	}
	if len(a.env) > 0 {
		cmdEnv, err := mergeEnv(os.Environ(), a.env)
		if err != nil {
			return "", fmt.Errorf("build %s env: %w", a.name, err)
		}
		cmd.Env = cmdEnv
	}
	var stderr strings.Builder
	cmd.Stderr = &stderr

	stdout, pipeErr := cmd.StdoutPipe()
	if pipeErr != nil {
		return "", fmt.Errorf("create codex stdout pipe: %w", pipeErr)
	}

	if err := cmd.Start(); err != nil {
		return "", fmt.Errorf("start codex: %w", err)
	}

	var threadID string
	var fallbackTexts []string
	scanner := bufio.NewScanner(stdout)
	scanner.Buffer(make([]byte, 0, 1024*1024), 1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		var event codexJSONEvent
		if err := json.Unmarshal([]byte(line), &event); err != nil {
			continue
		}
		if event.ThreadID != "" {
			threadID = event.ThreadID
		}
		if event.Type == "item.completed" && event.Item != nil && event.Item.Type == "agent_message" && event.Item.Text != "" {
			fallbackTexts = append(fallbackTexts, event.Item.Text)
		}
	}
	if err := scanner.Err(); err != nil {
		log.Printf("[cli] codex stdout scan error: %v", err)
	}

	err := cmd.Wait()
	if err != nil {
		errMsg := strings.TrimSpace(stderr.String())
		if errMsg != "" {
			return "", fmt.Errorf("codex error: %w, stderr: %s", err, errMsg)
		}
		return "", fmt.Errorf("codex error: %w", err)
	}

	if threadID != "" {
		a.mu.Lock()
		a.sessions[conversationID] = threadID
		a.mu.Unlock()
		sessionPath, err := a.saveSession(conversationID, threadID)
		if err != nil {
			log.Printf("[cli] failed to persist codex thread (thread=%s, conversation=%s): %v", threadID, conversationID, err)
		} else {
			log.Printf("[cli] persisted codex thread (thread=%s, conversation=%s, path=%s)", threadID, conversationID, sessionPath)
		}
		log.Printf("[cli] saved codex thread (thread=%s, conversation=%s)", threadID, conversationID)
	}

	var result string
	if data, readErr := os.ReadFile(outFile); readErr == nil && len(data) > 0 {
		result = string(data)
	} else if len(fallbackTexts) > 0 {
		result = strings.Join(fallbackTexts, "")
	}
	result = strings.TrimSpace(result)
	if result == "" {
		return "", fmt.Errorf("codex returned empty response")
	}
	return result, nil
}

type persistedCLISession struct {
	Agent          string `json:"agent"`
	Command        string `json:"command"`
	ConversationID string `json:"conversation_id"`
	SessionID      string `json:"session_id"`
}

func (a *CLIAgent) sessionPath(conversationID string) string {
	home, err := os.UserHomeDir()
	if err != nil {
		return filepath.Join(os.TempDir(), "weclaw-sessions", a.sessionFileName(conversationID))
	}
	return filepath.Join(home, ".weclaw", "sessions", a.sessionFileName(conversationID))
}

func (a *CLIAgent) sessionFileName(conversationID string) string {
	sum := sha256.Sum256([]byte(a.name + "\x00" + conversationID))
	return fmt.Sprintf("%s-%x.json", safeSessionName(a.name), sum[:12])
}

func safeSessionName(name string) string {
	var b strings.Builder
	for _, r := range name {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '-' || r == '_' {
			b.WriteRune(r)
		}
	}
	if b.Len() == 0 {
		return "agent"
	}
	return b.String()
}

func (a *CLIAgent) loadSession(conversationID string) (string, bool) {
	path := a.sessionPath(conversationID)
	data, err := os.ReadFile(path)
	if err != nil {
		return "", false
	}
	var s persistedCLISession
	if err := json.Unmarshal(data, &s); err != nil {
		log.Printf("[cli] failed to parse session file (command=%s, conversation=%s): %v", a.command, conversationID, err)
		return "", false
	}
	if s.SessionID == "" {
		return "", false
	}
	log.Printf("[cli] loaded persisted session (agent=%s, session=%s, conversation=%s, path=%s)", a.name, s.SessionID, conversationID, path)
	return s.SessionID, true
}

func (a *CLIAgent) saveSession(conversationID, sessionID string) (string, error) {
	path := a.sessionPath(conversationID)
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return path, err
	}
	data, err := json.MarshalIndent(persistedCLISession{
		Agent:          a.name,
		Command:        a.command,
		ConversationID: conversationID,
		SessionID:      sessionID,
	}, "", "  ")
	if err != nil {
		return path, err
	}
	return path, os.WriteFile(path, data, 0o600)
}

func filterCodexResumeArgs(args []string) []string {
	filtered := make([]string, 0, len(args))
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--sandbox", "-s":
			if i+1 < len(args) {
				i++
			}
		case "--cd", "-C", "--add-dir":
			if i+1 < len(args) {
				i++
			}
		default:
			filtered = append(filtered, args[i])
		}
	}
	return filtered
}
