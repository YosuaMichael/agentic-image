# Security

- Never commit API keys, tokens, or credentials (`.env` is gitignored; only
  `.env.example` is tracked). Report accidental commits immediately so keys
  can be rotated.
- The artifact server (`scripts/serve_artifacts.py`) binds loopback only and
  path-constrains every request to its root. Remote access is delegated to a
  private tailnet (`tailscale serve`) — do not expose it to the public
  internet as-is. Use `--token` for an extra shared-secret layer.
- Model weights are fetched by each user from gated Hugging Face repos under
  the Ideogram 4 Non-Commercial license; this repo never redistributes them.
- To report a vulnerability, open a private security advisory on GitHub (or
  contact the maintainers) rather than filing a public issue.
