# Configuración

Qaithon necesita credenciales únicamente cuando querés usar un QPU cloud real.
Para los simuladores locales (`mock`, `ibm.aer`, `quandela.perceval`,
`pennylane.sim`, `deepquantum`) no se requiere ninguna configuración.

## Fuentes de credenciales, en orden de prioridad

1. **Setters programáticos** — funciones `qaithon.set_*` llamadas en Python.
2. **Variables de entorno** — exportadas en el shell.
3. **Archivo `.env`** — en la raíz del proyecto, cargado automáticamente.

La primera fuente que entrega un valor gana. Las variables de entorno
exportadas siempre tienen prioridad sobre el `.env`, y las llamadas a setters
sobre ambas.

## API programática (recomendada)

```python
import qaithon

qaithon.set_ibm_token("TU_API_KEY_DE_44_CHARS")
qaithon.set_aws_credentials("AKIA...", "secret...", region="us-east-1")
qaithon.set_quandela_token("TU_TOKEN_QUANDELA")
qaithon.set_huggingface_token("hf_...")
```

Setup multi-proveedor en un solo call:

```python
qaithon.configure(
    ibm_token="...",
    aws_access_key_id="AKIA...",
    aws_secret_access_key="...",
    quandela_token="...",
    huggingface_token="hf_...",
)
```

Verificá qué está configurado sin exponer los valores:

```python
qaithon.config.status()
# {'ibm': True, 'aws': True, 'quandela': True, 'huggingface': True}
```

## Variables de entorno

| Variable | Uso |
|---|---|
| `IBM_QUANTUM_TOKEN` | API key de IBM Cloud IAM. |
| `IBM_QUANTUM_CHANNEL` | Por defecto `ibm_quantum_platform`. |
| `IBM_QUANTUM_INSTANCE` | CRN opcional de una instancia específica. |
| `AWS_ACCESS_KEY_ID` | Access key de IAM (`AKIA...`). |
| `AWS_SECRET_ACCESS_KEY` | Secret de IAM. |
| `AWS_DEFAULT_REGION` | Por defecto `us-east-1`. |
| `QUANDELA_TOKEN` | Token de Quandela Cloud. |
| `HF_TOKEN` | Token de HuggingFace Hub (write recomendado). |

## Archivo `.env`

El archivo `.env` en la raíz del proyecto se carga automáticamente al
importar Qaithon. Ejemplo:

```bash
IBM_QUANTUM_TOKEN=...
IBM_QUANTUM_CHANNEL=ibm_quantum_platform

AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1

QUANDELA_TOKEN=...
HF_TOKEN=hf_...
```

Siempre agregá `.env` a tu `.gitignore`. El repo trae un `.env.example` como
plantilla.

## Cómo obtener cada credencial

### IBM Quantum

1. Cuenta en <https://quantum.cloud.ibm.com>.
2. Click en `Identity and Access Management` desde tu cuenta (panel derecho) —
   abre IBM Cloud IAM.
3. Crear API key. **Copiala una sola vez — no se vuelve a mostrar.**

Plan gratuito: 10 minutos de QPU por mes.

### AWS Braket

1. Cuenta AWS + activar Braket en la consola.
2. **IAM → Users → Create user** con la policy `AmazonBraketFullAccess`.
3. **Security credentials → Create access key** tipo **CLI**.
4. Copiá el Access Key ID y el Secret Access Key.

Free Tier incluye ~1 hora/mes del simulador SV1. QPUs reales (QuEra, IonQ)
son pay-per-shot.

### Quandela Cloud

1. Registrate en <https://cloud.quandela.com>.
2. Si tu cuenta es nueva, aplicá para acceso de research.
3. Generá un token desde el dashboard.

Free tier para cuentas de research aprobadas es generoso; el pricing de
producción no se lista públicamente.

### HuggingFace Hub

1. Cuenta en <https://huggingface.co>.
2. Crear organización llamada `qaithon` (o similar) para reservar namespace.
3. Generar token fine-grained en <https://huggingface.co/settings/tokens>
   con acceso **write** limitado a tu org.

Gratis para todos los casos de uso.

## Verificar el setup

```bash
qaithon doctor          # Inspecciona Python, PyTorch, backends, devices.
qaithon list-backends   # Muestra qué backends están utilizables ahora.
```

```python
import qaithon
qaithon.config.status()
```

Si un backend aparece como no disponible, verificá que la credencial
correspondiente esté configurada (o usá el simulador local equivalente).
