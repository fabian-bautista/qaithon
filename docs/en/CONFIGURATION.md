# Configuration

Qaithon needs credentials only when you want to use a real cloud QPU. For local
simulators (`mock`, `ibm.aer`, `quandela.perceval`, `pennylane.sim`,
`deepquantum`) no configuration is required at all.

## Credential sources, in order of precedence

1. **Programmatic setters** — `qaithon.set_*` functions called in Python.
2. **Environment variables** — exported in the shell.
3. **`.env` file** — at the project root, auto-loaded on import.

The first source that provides a value wins. Environment variables exported in
the shell always override the `.env` file, and explicit setter calls override
both.

## Programmatic API (recommended)

```python
import qaithon

qaithon.set_ibm_token("YOUR_44_CHAR_API_KEY")
qaithon.set_aws_credentials("AKIA...", "secret...", region="us-east-1")
qaithon.set_quandela_token("YOUR_QUANDELA_TOKEN")
qaithon.set_huggingface_token("hf_...")
```

One-shot multi-provider setup:

```python
qaithon.configure(
    ibm_token="...",
    aws_access_key_id="AKIA...",
    aws_secret_access_key="...",
    quandela_token="...",
    huggingface_token="hf_...",
)
```

Check what is wired up without exposing values:

```python
qaithon.config.status()
# {'ibm': True, 'aws': True, 'quandela': True, 'huggingface': True}
```

## Environment variables

| Variable | Used for |
|---|---|
| `IBM_QUANTUM_TOKEN` | IBM Cloud IAM API key. |
| `IBM_QUANTUM_CHANNEL` | Default `ibm_quantum_platform`. |
| `IBM_QUANTUM_INSTANCE` | Optional CRN of a specific instance. |
| `AWS_ACCESS_KEY_ID` | IAM access key (`AKIA...`). |
| `AWS_SECRET_ACCESS_KEY` | IAM secret. |
| `AWS_DEFAULT_REGION` | Default `us-east-1`. |
| `QUANDELA_TOKEN` | Quandela Cloud API token. |
| `HF_TOKEN` | HuggingFace Hub token (write recommended). |

## `.env` file

A `.env` file at the project root is auto-loaded on import. Example:

```bash
IBM_QUANTUM_TOKEN=...
IBM_QUANTUM_CHANNEL=ibm_quantum_platform

AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1

QUANDELA_TOKEN=...
HF_TOKEN=hf_...
```

Always add `.env` to your `.gitignore`. A `.env.example` template ships with
the repo.

## Where to get each credential

### IBM Quantum

1. Sign up at <https://quantum.cloud.ibm.com>.
2. Click `Identity and Access Management` from your account page (right side
   panel) — that opens IBM Cloud IAM.
3. Create an API key. **Copy it once — it will not be shown again.**

Free plan: 10 minutes of QPU time per month.

### AWS Braket

1. Create an AWS account and activate Braket in the AWS console.
2. Go to **IAM → Users → Create user** with the policy `AmazonBraketFullAccess`.
3. **Security credentials → Create access key** of type **CLI**.
4. Copy both the Access Key ID and the Secret Access Key.

Free Tier includes ~1 hour/month of SV1 simulator. Real QPUs (QuEra, IonQ)
are pay-per-shot.

### Quandela Cloud

1. Sign up at <https://cloud.quandela.com>.
2. Apply for research access if your account is new.
3. Generate a token from the dashboard.

Free tier for approved research accounts is generous; pricing for production
is not publicly listed.

### HuggingFace Hub

1. Create an account at <https://huggingface.co>.
2. Create an organization named `qaithon` (or similar) to reserve a namespace.
3. Generate a fine-grained token at <https://huggingface.co/settings/tokens>
   with **write** access scoped to your org.

Free for all use cases.

## Verifying the setup

```bash
qaithon doctor          # Inspects Python, PyTorch, backends, devices.
qaithon list-backends   # Shows which backends are usable right now.
```

```python
import qaithon
qaithon.config.status()
```

If a backend appears unavailable, check the corresponding credential is set
(or run the local-simulator equivalent).
