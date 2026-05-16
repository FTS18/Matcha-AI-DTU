# Security Hardening Protocol

## Security Philosophy
Matcha AI adopts a "Security-by-Design" approach, focusing on the mitigation of common web vulnerabilities (OWASP Top 10) within a distributed AI context.

## Implemented Mitigations

### 1. Log Injection (CWE-117)
To prevent attackers from manipulating system logs or injecting malicious control characters, a mandatory sanitization layer is applied to all logging operations.
*   **Mechanism**: The `_sanitize` helper strips `\r` and `\n` characters from dynamic strings.
*   **Implementation Locations**: `services/inference/app/api/routes_stub.py`, `services/inference/app/core/goal_detection.py`.

### 2. Path Traversal (CWE-22)
File system interactions in the orchestrator are protected against directory traversal attacks.
*   **Normalization**: Uses `path.resolve()` to generate canonical paths.
*   **Validation**: Ensures all file operations are strictly confined to the `/uploads/` directory using `startsWith()` boundary checks.
*   **Isolation**: Randomized UUID prefixes are applied to all user-uploaded content to prevent filename collision and predictability.

### 3. Server-Side Request Forgery (SSRF)
Vulnerabilities in the WebSocket upgrade mechanism within the Next.js framework were mitigated by reconciling the project dependencies to a patched version.
*   **Version Baseline**: Next.js 14.2.15+.

### 4. Secret Redaction
The system is configured to prevent the accidental leakage of sensitive credentials (e.g., Roboflow API keys, database URLs) in error logs or standard output.
*   **Error Masking**: Initialization exceptions are intercepted to redact configuration details before being emitted to the logging stream.

## Ongoing Monitoring
We recommend regular static analysis scans using CodeQL and dependency audits via `npm audit` and `Safety` (for Python) to maintain this security posture.
