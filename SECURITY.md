# Security Policy — Losion

> **Losion** is an open-source Python/PyTorch AI framework for building
> advanced reasoning, retrieval-augmented, and elastic-capacity language models.
> We take the security of our project and its users seriously.

---

## 1. Overview

This document outlines the security policy for the Losion project. It describes
which versions are currently supported, how to responsibly report
vulnerabilities, our response timelines, our disclosure policy, and security
best practices for users working with AI models and training data.

We encourage responsible disclosure and are committed to working with the
community to resolve security issues promptly and transparently.

---

## 2. Supported Versions

| Version | Branch / Tag        | Status       | Support Ends |
|---------|---------------------|--------------|--------------|
| 2.x     | `main`              | ✅ Active    | Next major   |
| 1.9.x   | `v1.9.x`            | ⚠️ Patch-only| EOL on 3.0   |
| ≤ 1.8   | —                   | ❌ End-of-life| —            |

> **Note:** Only the latest minor release on `main` receives active feature and
> security updates. Prior minor versions receive critical patches only for 30
> days after a new minor is published.

---

## 3. Reporting a Vulnerability

### 3.1 How to Report

We strongly prefer vulnerability reports to be submitted **privately** so that
we can fix the issue before it becomes publicly known.

| Method                        | Details                                            |
|-------------------------------|----------------------------------------------------|
| **GitHub Security Advisory**  | [Open a private advisory](https://github.com/Wolfvin/Losion/security/advisories/new) |
| **Email**                     | security@losion.dev                                |

Please **do not** file security vulnerabilities as public GitHub issues.

### 3.2 What to Include

To help us triage and resolve the issue quickly, please include:

1. **Description** — A clear description of the vulnerability.
2. **Affected versions** — Which versions or commits are affected.
3. **Reproduction steps** — Minimal code or commands to reproduce the issue.
4. **Impact** — What an attacker could achieve (e.g., code execution, data
   leak, model poisoning).
5. **Suggested fix** — If you have a patch or mitigation, feel free to share it.

### 3.3 What to Expect

You will receive an acknowledgment of your report within **48 hours**. We will
keep you updated on our progress and coordinate the public disclosure with you.

---

## 4. Response Timeline

| Phase                     | Target Time        |
|---------------------------|--------------------|
| Acknowledgment of report  | ≤ 48 hours         |
| Initial triage & severity | ≤ 5 business days  |
| Fix developed & verified  | ≤ 14 business days |
| Patch released            | ≤ 21 business days |
| Public advisory published | With patch release |

> Critical vulnerabilities (remote code execution, credential theft, model
> supply-chain attacks) are prioritized and may be resolved on an accelerated
> timeline.

If we are unable to meet these timelines, we will communicate the delay and
provide an estimated resolution date.

---

## 5. Disclosure Policy

- **Coordinated disclosure** — We work with reporters to publish advisories
  simultaneously with (or shortly after) the patched release.
- **Embargo period** — We may request a short embargo (up to 30 days) to allow
  downstream users to patch before full public disclosure.
- **Credit** — Reporters who follow responsible disclosure are credited in the
  advisory (unless they prefer to remain anonymous).
- **No bounty program** — Losion is a community project and does not currently
  offer monetary bounties, but we deeply appreciate responsible disclosure.

---

## 6. Security Best Practices for Users

Losion is an AI framework that involves model training, inference, and data
processing. Follow these guidelines to use Losion securely.

### 6.1 Model Safety

- **Verify checkpoints** — Only load model weights (`.pt`, `.pth`, `.safetensors`)
  from trusted sources. Use `torch.load(..., weights_only=True)` when possible
  to prevent arbitrary code execution via pickled tensors.
- **Safetensors preferred** — Prefer the Safetensors format over PyTorch's
  default pickle-based format to eliminate deserialization attacks.
- **Model provenance** — Record and verify checksums (SHA-256) of downloaded
  checkpoints before loading them.
- **Sandbox inference** — Run inference on untrusted models in a sandboxed or
  containerized environment.

### 6.2 Data Privacy

- **Sanitize training data** — Audit datasets for personally identifiable
  information (PII) before training. Use the `lossion.utils.logging` module's
  redaction features where applicable.
- **No data in repos** — Never commit raw data files, user data, or model
  weights to version control. Use the provided `.gitignore` patterns.
- **Secure data pipelines** — Use encrypted storage and transfer for sensitive
  datasets. Avoid logging raw inputs to disk or external services.

### 6.3 Environment & Dependencies

- **Pin dependencies** — Use `pip install -r requirements.txt` with pinned
  versions. Regularly audit with `pip-audit` or `safety`.
- **Virtual environments** — Always install Losion in an isolated virtual
  environment; never install globally or in system Python.
- **Keep updated** — Upgrade to the latest supported version promptly to
  receive security patches.

### 6.4 Training Infrastructure

- **WandB / MLflow** — If using experiment tracking, ensure API keys are stored
  in environment variables or a secrets manager — never in code or config files
  committed to Git.
- **Distributed training** — Secure communication between nodes with TLS and
  authentication. Do not expose `torch.distributed` ports publicly.
- **GPU access** — Restrict GPU access to authorized users. Unrestricted GPU
  compute can be abused for cryptomining or denial-of-service.

### 6.5 Supply-Chain Security

- **Pre-commit hooks** — Use the project's `.pre-commit-config.yaml` to
  enforce linting, formatting, and secret detection before commits.
- **Dependency review** — Review all third-party packages added to the
  project. Prefer well-maintained, widely-used libraries.
- **CI/CD hardening** — Pin action versions in GitHub Actions workflows and
  minimize the use of third-party actions.

---

## 7. Contact

| Role                | Contact                        |
|---------------------|--------------------------------|
| Security Team       | security@losion.dev            |
| Project Maintainer  | See GitHub repository admins   |
| General Questions   | GitHub Discussions             |

---

_Thank you for helping keep Losion and its community safe._
