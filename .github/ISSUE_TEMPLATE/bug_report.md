---
name: Bug Report
about: Create a report to help us improve AI Agent HA
title: '[BUG] '
labels: bug
assignees: ''

---

## Bug Description
A clear and concise description of what the bug is.

## Environment
- **Home Assistant version**: [e.g., 2023.12.0]
- **AI Agent HA version**: [e.g., 1.0.0]
- **AI Provider**: [e.g., OpenAI, Gemini, Anthropic, etc.]
- **AI Model**: [e.g., gpt-4, claude-3-sonnet, etc.]
- **Installation method**: [HACS/Manual]

## Steps to Reproduce
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '...'
3. Enter prompt '...'
4. See error

## Expected Behavior
A clear and concise description of what you expected to happen.

## Actual Behavior
A clear and concise description of what actually happened.

## Screenshots/Logs
If applicable, add screenshots or Home Assistant logs to help explain your problem.

### How to Collect Logs:

1. **Enable debug logging** by adding this to your `configuration.yaml`:
   ```yaml
   logger:
     default: info
     logs:
       custom_components.ai_agent_ha: debug
   ```

2. **Restart Home Assistant** to apply logging changes

3. **Reproduce the issue** to capture it in the logs

4. **Collect logs** using one of these methods:
   - **UI Method**: Go to Settings → System → Logs → Load Full Logs → Search for "ai_agent_ha"
   - **File Method**: Check `<config_dir>/home-assistant.log` for entries containing "ai_agent_ha"

**Note**: API keys and tokens are automatically sanitized in debug logs and won't be exposed.

For detailed instructions, see our [Troubleshooting Guide](https://github.com/sbenodiz/ai_agent_ha/blob/main/docs/TROUBLESHOOTING.md#collecting-and-sharing-logs).

### Logs:
```
[Paste relevant logs here]
```

## Configuration
Please provide your configuration (remove sensitive information like API keys):

```yaml
# Your AI Agent HA configuration
```

## Additional Context
Add any other context about the problem here.

## Checklist
- [ ] I have searched existing issues to avoid duplicates
- [ ] I am using the latest version of AI Agent HA
- [ ] I have tested with multiple AI providers (if applicable)
- [ ] I have included all relevant logs and configuration
- [ ] I have removed sensitive information from configuration (API keys are auto-sanitized in logs) 