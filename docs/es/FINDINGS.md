# Qué hace Qaithon de verdad — hallazgos de investigación

Este documento reporta, de forma honesta y reproducible, qué construimos y
medimos: **correr partes de redes neuronales reales con algoritmos cuánticos y
fotónicos genuinos**, y exactamente hasta dónde llega la tecnología de hoy. Todo
aquí es reproducible en un laptop — los scripts están en `scripts/`.

> **Contrato de honestidad.** "Cómputo" significa que el matmul lo evalúa un
> algoritmo cuántico/fotónico *real* (Perceval/MerLin en fotónica,
> Qiskit/PennyLane en qubits) — nunca matemática clásica con etiqueta cuántica.
> Todos los resultados de abajo corrieron en **simuladores**, no en una QPU
> física. El mismo código apunta a hardware real cuando se vuelva accesible.

---

## En lenguaje simple (sin jerga)

Nos hicimos una sola pregunta: *¿puede esta librería correr cómputo de IA real
usando algoritmos cuánticos (qubits) y fotónicos (luz) — de verdad, no fingido?*

La respuesta es **sí, a escala diminuta, y lo probamos:**

- Un **modelo de lenguaje real y preentrenado** (TinyStories-1M) escribió un
  **cuento coherente** — *"...una niña llamada Lily a la que le gustaba jugar al
  sol..."* — con su matemática computada por **circuitos cuánticos genuinos**
  (simulados). La salida fue **idéntica** al resultado clásico normal.
- Capas neuronales diminutas **entrenan y corren** con algoritmos tanto
  fotónicos como cuánticos.
- **Medimos los límites exactos** de cada tecnología en un laptop normal.

Es pequeño — pero es **real**, **honesto**, y **cualquiera puede reproducirlo**.

---

## Resultados destacados (los números)

| Qué | Resultado |
|---|---|
| Matmul fotónico genuino vs clásico | error **~1e-7** (exacto) |
| Matmul cuántico genuino vs clásico | error **~1e-7** (exacto) |
| **TinyStories-1M, inferencia cuántica genuina** | **texto coherente**, 48 capas genuinas, fidelidad vs clásico **1.08e-6**, argmax **100%** idéntico |
| Capa fotónica diminuta (entrenable) | aprende una tarea de juguete (~0.83 acc) |
| Capa cuántica diminuta (entrenable) | aprende una tarea de juguete (~0.95 acc) |

Cómo funciona el matmul genuino: cualquier matriz de pesos `W` se embebe en una
*unitaria* (dilatación de Halmos), se realiza como un circuito óptico-lineal real
(fotónica) o un circuito de qubits (cuántica), la entrada se codifica en amplitud
y se leen las amplitudes de salida. La comparación clásica solo se usa para
*verificar* que el circuito hizo la cuenta.

---

## Entender los límites (esta es la parte más útil)

¿Por qué solo corre modelos *diminutos*? La razón es distinta — y **opuesta** —
para cada tecnología.

### Fotónica — "un casillero por número"

La fotónica guarda cada número en un **carril de luz (un modo)**: aproximadamente
**un carril por número**. El simulador (Perceval) maneja hasta **256 carriles → ~128
números de ancho** por capa.

¿Por qué ese tope? Un simulador existe para **imitar un chip real**, y los chips
fotónicos reales son *diminutos*: Quandela Belenos tiene **12 modos**; los de
investigación, unas pocas decenas. Así que 256 ya es ~20× más grande que
cualquier cosa que exista físicamente. Pasarse sería o bien (a) una
multiplicación de matrices clásica común (la óptica de un fotón es
matemáticamente solo matriz×vector — sin valor cuántico), o bien (b) chocar con la
explosión combinatoria de la simulación multi-fotón. El tope está justo en el
borde de lo que tiene sentido.

### Cuántica — "cada qubit duplica el cupo" (y es al revés)

Los qubits guardan **2ⁿ números con n qubits** — exponencial. 9 qubits guardan
512 números; 11 qubits guardan 2048. Así que una capa de ancho 768 (GPT-2)
necesita solo **11 qubits**.

Aquí la situación es **opuesta a la fotónica**: el *hardware real* es enorme
comparado con lo que cualquier computadora clásica puede simular.

| | Hardware real | Simulador clásico |
|---|---|---|
| Fotónica | ~12 modos | 256 modos → el simulador **sobra** |
| Cuántica | 156+ qubits físicos (IBM); 48–94 lógicos (Quantinuum) | ~30 qubits (laptop) → el simulador **se queda corto** |

Un simulador de statevector guarda 2ⁿ números, y **cada qubit duplica la
memoria**: 30 qubits ≈ 17 GB (llena un laptop de 16 GB), 50 qubits es una
supercomputadora récord mundial, y 156 qubits necesitarían más números que átomos
hay en el universo. **No se puede simular clásicamente una computadora cuántica
grande — y esa imposibilidad es exactamente por qué las computadoras cuánticas
son valiosas.**

### Frontera medida (MacBook Air M2, 16 GB)

| Tecnología | Genuino + rápido | Usable | Tope duro |
|---|---|---|---|
| Fotónica (Perceval) | dim ≤ 128 (<0.1s) | — | **dim 128** (tope sim de 256 modos) |
| Cuántica (Qiskit statevector) | dim ≤ 256 (<0.05s) | dim ≤ 1024 (~2s) | dim ~4096 (~3 min — costo de la unitaria densa) |

**En una línea:** la fotónica se queda sin *carriles* (lineal, ~128); la cuántica
es tan poderosa que no se puede *simular* grande (exponencial) — pero con pocos
qubits empaqueta muchísimo más, así que alcanza capas más anchas. Por eso
TinyStories (una capa interna de ancho 256) corrió en cuántica pero no en fotónica.

---

## Qué corregimos en el camino (honestidad total)

Construir esto sacó a la luz problemas reales; los arreglamos y los registramos:

- **Inferencia y entrenamiento son dos caminos genuinos distintos.** El kernel de
  dilatación corre **inferencia** genuina de pesos arbitrarios/preentrenados pero
  *no es diferenciable*; el **entrenamiento** usa las capas diferenciables de
  MerLin/PennyLane.
- **La fotónica no puede correr un transformer completo genuino** — sus
  proyecciones internas (p. ej. atención QKV = 3× el ancho, MLP = 4× el ancho)
  exceden el techo fotónico de 128 de ancho. El camino fotónico genuino es la
  *capa entrenable PhotonicLayer*, no compilar transformers.
- Se arregló un bug de dispositivo (tensores CPU↔Metal) para que los modelos en
  Apple Silicon funcionen.

---

## Reprodúcelo tú mismo

Todo está empaquetado. En Python 3.10+:

```bash
pip install -e ".[huggingface,pennylane,quandela,deepquantum]"

python scripts/run_all_experiments.py   # regenera la tabla completa de resultados
```

O directamente:

```python
import qaithon
from transformers import AutoModelForCausalLM

# Inferencia cuántica genuina de un modelo real preentrenado (TinyStories-1M)
model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")
model = qaithon.compile(model, backends=("pennylane.sim",))  # cuántico, genuino
# ...generá como siempre; las capas lineales corren en un circuito de qubits real.
```

No necesitas cuenta de hardware cuántico — los algoritmos genuinos corren en
simuladores locales.

---

## Por qué importa, y qué viene

Hoy, Qaithon es dos cosas a la vez:

1. **Un instrumento genuino** para correr, entrenar y medir capas neuronales
   diminutas con algoritmos cuánticos y fotónicos reales — y para entender, con
   números medidos, exactamente dónde está parada la tecnología.
2. **Plomería lista para el futuro.** La misma llamada `qaithon.compile(model)`
   que hoy apunta a un simulador apuntará a una QPU real mañana — solo cambia un
   flag de backend.

Y el futuro no es hipotético. El hardware corregido de errores con **qubits
lógicos ya existe** (la Helios de Quantinuum reporta hasta 48–94 qubits lógicos);
la barrera es **acceso y costo**, no física — y ambos están mejorando. El día que
ese hardware se abra, el trabajo de aquí rinde directo: los kernels, los guards de
tamaño y el ruteo por tecnología ya están construidos. Qaithon se vuelve mucho más
que un envoltorio de simuladores.

**La invitación:** clónalo, córrelo, empuja los límites y prepárate. Este es un
primer paso pequeño y honesto en un camino que el hardware está construyendo
activamente.
