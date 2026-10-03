"""
Reconstruye fuente/presentacion.html a partir de las cifras de figuras/resultados.json.
Uso:  python scripts/construir_presentacion.py
Luego se imprime a PDF con Chrome (tamaño de página 1280 x 720).
"""
import json
from pathlib import Path

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "fuente" / "presentacion.html"
R = json.loads((ROOT / "figuras" / "resultados.json").read_text(encoding="utf-8"))
old = path.read_text(encoding="utf-8")
head = old[: old.index("<body>") + len("<body>")]
EXTRA_CSS = """
  .steps { display: flex; flex-direction: column; gap: 10px; }
  .step { display: flex; gap: 14px; align-items: flex-start; background: #fff; border: 1px solid var(--line); border-radius: 12px; padding: 12px 16px; }
  .step .n { flex: none; width: 34px; height: 34px; border-radius: 50%; background: var(--navy); color: #fff; font-weight: 700; font-size: 18px;
             display: flex; align-items: center; justify-content: center; }
  .step p { font-size: 18px; line-height: 1.35; }
  .bigq { font-size: 26px; line-height: 1.4; color: var(--navy); font-weight: 600; background: var(--soft); border-radius: 14px; padding: 18px 26px; margin-bottom: 18px; }
"""
if ".steps {" not in head:
    head = head.replace("</style>", EXTRA_CSS + "</style>", 1)
fmt = HtmlFormatter(noclasses=True, style="github-dark", nowrap=True)


def code(src, size=15):
    return f'<pre class="code" style="font-size:{size}px">{highlight(src.strip(), PythonLexer(), fmt)}</pre>'


def p(v, d=1):
    return f"{v * 100:.{d}f}".replace(".", ",") + " %"


def num(v, d=1):
    return f"{v:.{d}f}".replace(".", ",")


def miles(v):
    return f"{v:,.0f}".replace(",", ".")


def steps(items):
    return '<div class="steps">' + "".join(
        f'<div class="step"><div class="n">{i}</div><p>{t}</p></div>' for i, t in enumerate(items, 1)) + "</div>"


def slide(kicker, title, body):
    return f'''
<section class="slide">
  <div class="kicker">{kicker}</div>
  <h1>{title}</h1>
  {body}
  <div class="foot"><span>Conformal Prediction</span><span class="pg"></span></div>
</section>'''


rg, fr = R["regresion"], R["fraude"]
T = {r["regla"]: r for r in fr["reglas_test"]}
PR = {r["regla"]: r for r in fr["reglas_produccion"]}
t5, sp, mo, pm = T["Umbral 0,5"], T["Split conformal 95 %"], T["Mondrian 95 %"], PR["Mondrian 95 %"]
cond = rg["cobertura_condicional"]
ck = list(cond)
narrow = 1 - rg["ancho_medio_cqr"] / rg["ancho_split"]
var = R["variabilidad"]
curve = fr["curva_mondrian"]

CORE = '''
import numpy as np

def conformal_quantile(scores, alpha):
    n = len(scores)                           # cuántos errores tengo
    k = int(np.ceil((n + 1) * (1 - alpha)))   # qué posición del ranking tomo
    if k > n:                                 # muy pocos datos:
        return np.inf                         #   no puedo prometer nada
    return np.sort(scores)[k - 1]             # ordeno y devuelvo el k-ésimo
'''
CORE_USE = '''
errores = np.array([0.3, 1.2, 0.7, 2.5, 0.1, 0.9, 1.8, 0.4, 3.1])
conformal_quantile(errores, alpha=0.2)   # 9 errores, quiero 80 %
# k = (9 + 1) x 0.8 = 8  ->  el 8.º más pequeño  ->  2.5
'''
SPLIT = '''
from xgboost import XGBRegressor

# Paso 1. Entreno el modelo (solo con train)
model = XGBRegressor(max_depth=3).fit(X_train, y_train)

# Paso 2. Mido sus errores en calibración
errores = np.abs(y_cal - model.predict(X_cal))

# Paso 3. Saco el umbral q_hat
q_hat = conformal_quantile(errores, alpha=0.10)

# Paso 4. Predicción ± q_hat
pred = model.predict(X_new)
lower, upper = pred - q_hat, pred + q_hat
'''
CQR = '''
# Paso 1. Dos XGBoost: uno para el "piso" y otro para el "techo"
q = dict(objective="reg:quantileerror", max_depth=3)
piso  = XGBRegressor(quantile_alpha=0.05, **q).fit(X_train, y_train)
techo = XGBRegressor(quantile_alpha=0.95, **q).fit(X_train, y_train)

# Paso 2. Error = cuánto se sale el dato real de la banda
errores = np.maximum(piso.predict(X_cal) - y_cal,
                     y_cal - techo.predict(X_cal))

# Paso 3. Saco el umbral q_hat
q_hat = conformal_quantile(errores, alpha=0.10)

# Paso 4. Corrijo la banda con q_hat
lower = piso.predict(X_new) - q_hat
upper = techo.predict(X_new) + q_hat
'''
CLS_A = '''
# Paso 1. Entreno el XGBoost del notebook 02 (solo con train)
clf = XGBClassifier(n_estimators=300, max_depth=6,
                    scale_pos_weight=spw,
                    enable_categorical=True).fit(X_train, y_train)

# Paso 2. Mido el error en cada transacción de calibración
P_cal = clf.predict_proba(X_cal)    # 2 columnas: [p(legítima), p(fraude)]

# de cada fila tomo la probabilidad de su clase REAL...
p_real = P_cal[np.arange(len(y_cal)), y_cal]

# ...y el error es lo que le faltó para llegar a 1
s_cal = 1 - p_real
'''
CLS_B = '''
# Paso 3. Saco el umbral q_hat
q_unico   = conformal_quantile(s_cal, 0.05)               # MÉTODO 1: con todos
q_legit   = conformal_quantile(s_cal[y_cal == 0], 0.05)   # MÉTODO 3: solo legítimas
q_fraude  = conformal_quantile(s_cal[y_cal == 1], 0.05)   # MÉTODO 3: solo fraudes

# Paso 4. Una etiqueta entra al conjunto si su error (1 - p) es <= su umbral
P = clf.predict_proba(X_test)
conjunto_m1 = (1 - P) <= q_unico                 # mismo umbral para las 2 etiquetas
conjunto_m3 = (1 - P) <= [q_legit, q_fraude]     # un umbral para cada etiqueta

# Decisión: alerto si "fraude" (columna 1) quedó dentro del conjunto
alerta = conjunto_m3[:, 1]
recall_score(y_test, alerta), precision_score(y_test, alerta)
'''
MAPIE = '''
# pip install mapie xgboost
from mapie.regression import SplitConformalRegressor, ConformalizedQuantileRegressor
from mapie.classification import SplitConformalClassifier

# MÉTODO 1 en regresión: banda fija
reg = SplitConformalRegressor(model, confidence_level=0.90, prefit=True)
reg.conformalize(X_cal, y_cal)                      # calibrar
y_pred, intervalos = reg.predict_interval(X_test)   # predecir con intervalo

# MÉTODO 2: CQR, con los XGBoost [piso, techo, mediana]
cqr = ConformalizedQuantileRegressor([piso, techo, mediana],
                                     confidence_level=0.90, prefit=True)
cqr.conformalize(X_cal, y_cal)
y_pred, intervalos = cqr.predict_interval(X_test)

# MÉTODO 1 en clasificación: conjuntos con un umbral único
cls = SplitConformalClassifier(clf, confidence_level=0.95, prefit=True)
cls.conformalize(X_cal, y_cal)
y_pred, conjuntos = cls.predict_set(X_test)
'''

S = []
S.append('''
<section class="slide cover">
  <div class="kicker" style="color:#8fd6f5">Aprendizaje Automático · MCDA 2026-2 · Línea Sistemas</div>
  <h1>Conformal Prediction</h1>
  <div class="sub">Cómo ponerle un margen de seguridad con garantía a cualquier modelo</div>
  <div class="meta">Pedro Villegas Michel<br>Profesor: Andrés Vásquez Restrepo · Universidad EAFIT<br>Octubre 2026</div>
  <div class="foot"><span>Repositorio: github.com/PedroV27/Conformal-Predictions</span><span class="pg"></span></div>
</section>''')

S.append(slide("Hoja de ruta · 20 minutos", "Agenda", '''
  <div class="grid2" style="gap:28px">
    <ul class="big" style="font-size:26px">
      <li><b>El problema:</b> un número no basta</li>
      <li><b>La idea,</b> con una analogía</li>
      <li><b>Por qué funciona</b> y los 4 pasos</li>
      <li><b>La función central</b> en Python</li>
    </ul>
    <ul class="big" style="font-size:26px">
      <li><b>Tres métodos,</b> paso a paso y comparados</li>
      <li><b>Ejemplo:</b> el caso de fraude del curso con XGBoost</li>
      <li><b>Riesgos</b> y cómo usarlo con MAPIE</li>
      <li><b>Aplicaciones</b> reales y en mi trabajo</li>
    </ul>
  </div>'''))

S.append(slide("El problema", "Un modelo nos da un número… pero ¿cuánto confiar en él?", '''
  <div class="grid2">
    <div>
      <div class="card" style="margin-bottom:18px"><h2>Lo que entrega hoy el modelo</h2>
        <p>“El gasto de marzo será <b>$1.200 M</b>”<br>“Esta transacción es fraude con <b>p = 0,83</b>”</p></div>
      <div class="card accent"><h2>Lo que necesita quien decide</h2>
        <p>“El gasto estará entre <b>$1.050 M y $1.380 M</b>, y acierto 9 de cada 10 veces”<br>
        “Con esta regla detecto <b>al menos el 95 % de los fraudes</b>”</p></div>
    </div>
    <ul style="margin-top:6px">
      <li>Un número solo <b>no dice qué tan lejos</b> puede estar la realidad.</li>
      <li>Las probabilidades de XGBoost sirven para <b>ordenar</b> los casos, pero no son confiables como porcentaje (lo vimos en el notebook 02).</li>
      <li>Necesitamos un <b>margen de seguridad</b> que se pueda comprobar.</li>
    </ul>
  </div>'''))

S.append(slide("La idea, con una analogía", "El empleado nuevo que estima facturas", '''
  <div class="grid2" style="grid-template-columns: 1.15fr 1fr; gap:32px">
    <div>''' + steps([
        "Le muestro <b>100 facturas cuyo valor real ya conozco</b> y que él nunca ha visto.",
        "Anoto <b>cuánto se equivocó</b> en cada una.",
        "Ordeno los errores de menor a mayor y miro el <b>error número 90</b>: digamos, $80 mil.",
        "Desde hoy, si dice “vale $1 millón”, yo digo: <b>“entre $920 mil y $1.080.000”</b>, y acierto el 90 % de las veces.",
    ]) + '''
    </div>
    <div>
      <table class="small">
        <tr><th>En la analogía</th><th>Nombre técnico</th></tr>
        <tr><td>El empleado</td><td>El <b>modelo</b> (XGBoost)</td></tr>
        <tr><td>Las 100 facturas</td><td>Set de <b>calibración</b></td></tr>
        <tr><td>Cuánto se equivocó</td><td>El <b>score</b> (error)</td></tr>
        <tr><td>El error número 90</td><td><b>q̂</b>, el umbral</td></tr>
        <tr><td>El 90 %</td><td><b>1 − α</b> (α = error que acepto)</td></tr>
      </table>
      <div class="take" style="font-size:18px">No necesito saber <b>cómo piensa</b> el empleado. Solo medí sus errores en casos que no había visto.</div>
    </div>
  </div>'''))

S.append(slide("La promesa", "Qué garantiza conformal prediction", '''
  <div class="formula">P( valor real ∈ rango que entrego ) ≥ 1 − α</div>
  <div class="grid3">
    <div class="card"><h2>Sin supuestos de distribución</h2><p>No importa si los datos son normales, sesgados o con valores extremos.</p></div>
    <div class="card"><h2>Para cualquier modelo</h2><p>Regresión logística, XGBoost o una red neuronal: se usa como <b>caja negra</b>.</p></div>
    <div class="card"><h2>Con pocos o muchos datos</h2><p>La garantía vale con los datos que tengo, no solo “con infinitos datos”.</p></div>
  </div>
  <div class="take">Un solo requisito: que los datos nuevos <b>se parezcan</b> a los de calibración (técnicamente, que sean <b>intercambiables</b>). Vovk, Gammerman &amp; Shafer (2005).</div>'''))

S.append(slide("Por qué funciona", "Un argumento de ranking", '''
  <svg viewBox="0 0 1136 170" width="1136" height="170" role="img" aria-label="Errores de calibración ordenados con el umbral marcado">
    <line x1="30" y1="95" x2="1100" y2="95" stroke="#c3c2b7" stroke-width="2"/>
    <g fill="#2a78d6">
      <circle cx="60" cy="95" r="10"/><circle cx="150" cy="95" r="10"/><circle cx="225" cy="95" r="10"/><circle cx="300" cy="95" r="10"/>
      <circle cx="390" cy="95" r="10"/><circle cx="455" cy="95" r="10"/><circle cx="540" cy="95" r="10"/><circle cx="620" cy="95" r="10"/>
      <circle cx="700" cy="95" r="10"/><circle cx="790" cy="95" r="10"/><circle cx="880" cy="95" r="10"/>
    </g>
    <circle cx="1010" cy="95" r="10" fill="#9ec5f4"/>
    <line x1="840" y1="30" x2="840" y2="140" stroke="#d03b3b" stroke-width="3"/>
    <text x="840" y="22" text-anchor="middle" font-size="19" fill="#d03b3b" font-weight="700">q̂: el umbral</text>
    <circle cx="745" cy="95" r="13" fill="#fcfcfb" stroke="#eb6834" stroke-width="4"/>
    <text x="745" y="160" text-anchor="middle" font-size="18" fill="#eb6834" font-weight="700">caso nuevo</text>
    <text x="30" y="160" font-size="18" fill="#52514e">error pequeño</text>
    <text x="1100" y="160" text-anchor="end" font-size="18" fill="#52514e">error grande</text>
  </svg>
  <ul style="margin-top:18px">
    <li>Pongo en fila los errores de calibración, de menor a mayor.</li>
    <li>El caso nuevo es “uno más del grupo”: su error puede caer en <b>cualquier posición de la fila con la misma probabilidad</b>.</li>
    <li>Si pongo el umbral en la posición 90 de 100, solo hay un 10 % de probabilidad de que quede por encima. <b>Ese es todo el teorema.</b></li>
  </ul>'''))

S.append(slide("El algoritmo", "Siempre los mismos cuatro pasos", '''
  <div class="grid2" style="grid-template-columns: 1.1fr 1fr">
    <div>''' + steps([
        "<b>Separar</b> los datos en entrenamiento, <b>calibración</b> y prueba.",
        "<b>Entrenar</b> el modelo solo con los datos de entrenamiento.",
        "<b>Medir los errores</b> (scores) del modelo en calibración.",
        "<b>Tomar el umbral q̂</b> y usarlo para armar el rango de cada caso nuevo.",
    ]) + '''
    </div>
    <div>
      <div class="formula sm">q̂ = el error en la posición ⌈(n+1)(1−α)⌉</div>
      <p class="small"><b>¿Por qué n+1?</b> Porque el caso nuevo también entra en la fila: con 9 errores de calibración hay 10 posiciones posibles.</p>
      <p class="small muted" style="margin-top:10px">Resultado garantizado: la cobertura queda entre 1 − α y 1 − α + 1/(n+1).</p>
    </div>
  </div>
  <div class="take">Es barato: <b>un solo entrenamiento</b> y ordenar n números. Se agrega a cualquier modelo que ya tengamos.</div>'''))

S.append(slide("En código · la pieza central", "Todo el método cabe en una función", f'''
  <div class="grid2" style="grid-template-columns: 1.3fr 1fr; gap:28px">
    <div>{code(CORE)}{code(CORE_USE)}</div>
    <div>
      <ul class="small">
        <li>Recibe los <b>errores de calibración</b> y el error que acepto (α).</li>
        <li><b>Ordena</b> los errores y devuelve el de la posición k: ese es <b>q̂</b>.</li>
        <li>No mira el modelo ni supone nada sobre los datos.</li>
        <li><b>Es la misma función para los tres métodos.</b> Solo cambia qué llamamos “error”.</li>
      </ul>
    </div>
  </div>'''))

S.append(slide("Los tres métodos", "Lo único que cambia es qué llamamos “error”", '''
  <table>
    <tr><th style="width:24%">Método</th><th style="width:34%">¿Qué es el “error”?</th><th>¿Qué entrega?</th></tr>
    <tr><td><b>1. Split conformal</b><br><span class="muted">banda fija</span></td><td>La distancia entre el valor real y la predicción</td><td>Predicción ± q̂: <b>el mismo margen para todos</b></td></tr>
    <tr><td><b>2. CQR</b><br><span class="muted">banda adaptativa</span></td><td>Cuánto se sale el valor real de una banda “piso–techo”</td><td>Un margen <b>ancho donde hay duda y angosto donde no</b></td></tr>
    <tr><td><b>3. Mondrian</b><br><span class="muted">umbral por clase</span></td><td>1 − la probabilidad que el modelo le dio a la respuesta correcta, <b>medido por separado en cada clase</b></td><td>La garantía se cumple <b>dentro de cada clase</b> (por ejemplo, en los fraudes)</td></tr>
  </table>
  <div class="take">Vamos uno por uno: primero el código paso a paso y después la comparación de resultados.</div>'''))

S.append(slide("Método 1 · paso a paso", "Split conformal: predicción ± un margen fijo", f'''
  <div class="grid2" style="grid-template-columns: 1.25fr 1fr; gap:28px">
    <div>{code(SPLIT)}</div>
    <div>
      <ul class="small">
        <li>Son exactamente los <b>cuatro pasos</b> de antes.</li>
        <li>El “error” es <b>cuánto se equivocó</b> el modelo: |real − predicho|.</li>
        <li><b>q̂</b> es el error que solo se supera el 10 % de las veces.</li>
        <li>El intervalo final es la predicción <b>más y menos q̂</b>.</li>
      </ul>
      <div class="take" style="font-size:18px">Limitación: el margen es <b>igual para todos</b> los casos, sean fáciles o difíciles.</div>
    </div>
  </div>'''))

S.append(slide("Método 2 · paso a paso", "CQR: un margen que se adapta a cada caso", f'''
  <div class="grid2" style="grid-template-columns: 1.3fr 1fr; gap:28px">
    <div>{code(CQR, 14.5)}</div>
    <div>
      <ul class="small">
        <li>Entreno <b>dos XGBoost</b>: uno predice el “piso” (5 %) y otro el “techo” (95 %).</li>
        <li>Esa banda ya es ancha donde hay duda, pero <b>no trae garantía</b>.</li>
        <li>El “error” es <b>cuánto se sale</b> el dato real de la banda.</li>
        <li>Con q̂ <b>corrijo</b> la banda: la ensancho si cubría de menos o la estrecho si cubría de más.</li>
      </ul>
    </div>
  </div>'''))

S.append(slide("Comparación · métodos 1 y 2", "Mismos datos, dos márgenes distintos", f'''
  <div class="fig"><img src="../figuras/a_regresion_split_vs_cqr.png" alt="Intervalos de split conformal y CQR"></div>
  <table class="small" style="margin-top:8px">
    <tr><th>Método</th><th>¿Cumple el 90 %?</th><th>Ancho promedio del intervalo</th><th>Lectura</th></tr>
    <tr><td>1. Split conformal (banda fija)</td><td class="num">Sí: {p(rg["cobertura_split"])}</td><td class="num">{num(rg["ancho_split"], 2)}</td><td>Mismo ancho en todas partes</td></tr>
    <tr><td><b>2. CQR (banda adaptativa)</b></td><td class="num">Sí: {p(rg["cobertura_cqr"])}</td><td class="num"><b>{num(rg["ancho_medio_cqr"], 2)}</b></td><td><b>{p(narrow, 0)} más angosto</b>: se abre solo donde hace falta</td></tr>
  </table>'''))

S.append(slide("Advertencia clave", "La garantía es un <u>promedio</u>, no se cumple en cada zona", f'''
  <div class="grid2" style="grid-template-columns: 1.25fr 1fr">
    <div class="fig"><img src="../figuras/a_cobertura_condicional.png" alt="Cobertura por zona de x"></div>
    <div>
      <ul>
        <li>La banda fija acierta el <b>{p(cond[ck[0]]["split"], 0)}</b> donde hay poco ruido y solo el <b>{p(cond[ck[-1]]["split"], 0)}</b> donde hay mucho.</li>
        <li>En promedio da {p(rg["cobertura_split"], 0)}: <b>cumple, pero de forma desigual</b>.</li>
        <li>CQR sube la zona difícil a <b>{p(cond[ck[-1]]["cqr"], 0)}</b>.</li>
        <li>Si el grupo que me importa es una <b>clase</b> (como los fraudes), necesito el método 3.</li>
      </ul>
    </div>
  </div>'''))

S.append(slide("Clasificación", "Ahora la respuesta es un conjunto de etiquetas", '''
  <div class="bigq">El modelo ya no dice “es fraude” o “no es fraude”. Dice: <b>“estas son las respuestas que no puedo descartar”</b>.</div>
  <div class="grid3" style="margin-bottom:18px">
    <div class="card"><p class="set" style="color:var(--blue)">{legítima}</p><p>Seguro de que es legítima.<br><b>No alerto.</b></p></div>
    <div class="card"><p class="set" style="color:var(--orange)">{fraude} o {legítima, fraude}</p><p>No puedo descartar fraude.<br><b>Alerto para revisión.</b></p></div>
    <div class="card"><p class="set" style="color:var(--muted)">{ } vacío</p><p>Caso muy raro: ninguna respuesta convence.<br><b>Aquí no alerto.</b></p></div>
  </div>
  <p><b>Regla de decisión:</b> alerto si “fraude” está en el conjunto. Así puedo medir con las <b>métricas del curso</b>: recall, precisión y matriz de confusión.</p>'''))

S.append(slide("Método 3 · paso a paso (1 de 2)", "Primero: medir el error en cada transacción", f'''
  <div class="grid2" style="grid-template-columns: 1.15fr 1fr; gap:28px">
    <div>{code(CLS_A, 14.5)}</div>
    <div>
      <p class="small" style="margin-bottom:8px"><b>El error</b> es lo que le faltó al modelo para darle 100 % a la respuesta correcta. Tres casos de ejemplo:</p>
      <table class="small" style="font-size:16px">
        <tr><th>Era…</th><th>El modelo dijo p(fraude) =</th><th>Error</th></tr>
        <tr><td>Legítima</td><td class="num">0,001</td><td class="num"><b>0,001</b>: acertó</td></tr>
        <tr><td>Fraude</td><td class="num">0,98</td><td class="num"><b>0,02</b>: acertó</td></tr>
        <tr><td>Fraude</td><td class="num">0,10</td><td class="num"><b>0,90</b>: se equivocó</td></tr>
      </table>
      <div class="take" style="font-size:18px">Error cerca de <b>0</b> = el modelo acertó con seguridad. Error cerca de <b>1</b> = el modelo se equivocó.</div>
    </div>
  </div>'''))

S.append(slide("Método 3 · paso a paso (2 de 2)", "Después: un umbral para cada clase", f'''
  <div class="grid2" style="grid-template-columns: 1.5fr 1fr; gap:24px">
    <div>{code(CLS_B, 13.6)}</div>
    <div>
      <div class="card" style="margin-bottom:12px; padding:14px 18px"><h2>Método 1: un solo umbral</h2>
        <p class="small">q̂ = {num(fr["q_unico_95"], 3)}, calculado con las 60.000. Solo alerta si p(fraude) ≥ <b>{num(fr["umbral_split_95"], 3)}</b>: casi nunca.</p></div>
      <div class="card accent" style="padding:14px 18px"><h2>Método 3: Mondrian</h2>
        <p class="small">Legítimas: q̂ = {num(fr["q_legitimas_95"], 3)} ({miles(fr["legitimas_cal"])} casos).<br>Fraudes: q̂ = {num(fr["q_fraudes_95"], 3)} ({fr["fraudes_cal"]} casos).<br>Alerta si p(fraude) ≥ <b>{num(fr["umbral_mondrian_95"], 3)}</b>.</p></div>
    </div>
  </div>
  <div class="take" style="font-size:18px">Es el mismo código. La única diferencia: Mondrian calcula el umbral de los fraudes <b>solo con los fraudes</b>, y por eso es mucho menos exigente para alertar.</div>'''))

S.append(slide("Ejemplo · fraude con tarjeta (dataset del curso)", "El montaje del ejercicio", f'''
  <div class="grid2" style="grid-template-columns: 1fr 1.4fr">
    <div>''' + steps([
        f"<b>Datos:</b> 300.000 transacciones, solo {p(fr['tasa_fraude'], 2)} son fraude.",
        "<b>Reparto:</b> 180 mil para entrenar, 60 mil para calibrar y 60 mil para probar.",
        f"<b>Modelo:</b> el XGBoost del notebook 02 (PR-AUC {num(fr['pr_auc_test'], 3)}).",
        "<b>Promesa pedida:</b> 95 %.",
    ]) + f'''
    </div>
    <div>
      <div class="fig" style="max-height:310px"><img src="../figuras/b_scores_mondrian.png" alt="Errores de calibración por clase"></div>
      <ul class="small" style="margin-top:8px; font-size:16px">
        <li>Cada barra cuenta cuántas transacciones de calibración tuvieron ese error. La línea punteada es el umbral q̂.</li>
        <li><b>Legítimas:</b> casi todas en 0; el modelo las acierta fácil. Umbral muy pequeño.</li>
        <li><b>Fraudes:</b> la mayoría en 0, pero varios cerca de 1: fraudes que el modelo no vio. Umbral grande.</li>
      </ul>
    </div>
  </div>'''))

S.append(slide("Comparación · métodos 1 y 3 (datos de prueba)", "¿Cuántos fraudes detecta cada regla?", f'''
  <div class="fig" style="max-height:300px"><img src="../figuras/b_reglas_recall_precision.png" alt="Recall y precisión por regla"></div>
  <table class="small" style="margin-top:6px">
    <tr><th>Regla de decisión</th><th>Recall</th><th>Precisión</th><th>Alertas</th><th>¿Qué promete?</th></tr>
    <tr><td>Umbral 0,5 (sin conformal)</td><td class="num">{p(t5["recall"])}</td><td class="num">{p(t5["precision"])}</td><td class="num">{miles(t5["alertas"])}</td><td>Nada</td></tr>
    <tr><td>1. Split conformal 95 %</td><td class="num">{p(sp["recall"])}</td><td class="num">{p(sp["precision"])}</td><td class="num">{miles(sp["alertas"])}</td><td>95 % en el total ({p(fr["cobertura_global_split_95"])}), no en los fraudes</td></tr>
    <tr><td><b>3. Mondrian 95 %</b></td><td class="num"><b>{p(mo["recall"])}</b></td><td class="num">{p(mo["precision"])}</td><td class="num">{miles(mo["alertas"])}</td><td><b>Detectar ≥ 95 % de los fraudes</b></td></tr>
  </table>'''))

S.append(slide("¿Por qué pasa esto?", "El 99 % legítimo “tapa” a los fraudes", f'''
  <div class="grid2">
    <div class="card"><h2>Método 1: un solo umbral</h2>
      <p>Mezcla todos los errores. Como casi todo es legítimo, el umbral lo deciden las legítimas.</p>
      <p style="margin-top:10px">La promesa del 95 % se cumple <b>en el total</b>, pero solo se detecta el <b>{p(sp["recall"], 0)}</b> de los fraudes.</p></div>
    <div class="card accent"><h2>Método 3: Mondrian</h2>
      <p>Calcula un umbral <b>solo con los fraudes</b> y otro solo con las legítimas.</p>
      <p style="margin-top:10px">La promesa se cumple <b>en cada clase</b>: detecta el <b>{p(mo["recall"], 0)}</b> de los fraudes. Eso es una <b>garantía de recall</b>.</p></div>
  </div>
  <div class="take">El precio: más alertas ({miles(mo["alertas"])} frente a {miles(sp["alertas"])}) y menos precisión. A cambio, <b>casi ningún fraude se escapa</b>.</div>'''))

S.append(slide("Ejemplo · el precio de pedir más", "A más recall prometido, más alertas", f'''
  <div class="fig"><img src="../figuras/b_recall_vs_alpha.png" alt="Recall frente a nivel pedido y alertas"></div>
  <div class="take">Mondrian cumple lo que promete en cualquier nivel (sigue la diagonal). Pero prometer 99 % en vez de 95 % sube las alertas de <b>{p(curve["0.95"]["alertas"])}</b> a <b>{p(curve["0.99"]["alertas"])}</b> de las transacciones: unas <b>6 veces más</b>.</div>'''))

S.append(slide("Ejemplo · la prueba de fuego (meses futuros)", "¿La promesa se sostiene en el futuro?", f'''
  <div class="grid2" style="grid-template-columns: 1.7fr 1fr; gap:28px">
    <div class="fig"><img style="width:100%" src="../figuras/b_matrices_confusion.png" alt="Matrices de confusión en datos de prueba y en datos futuros"></div>
    <div>
      <ul class="small" style="font-size:17px">
        <li><b>Izquierda, datos de prueba:</b> 60.000 transacciones del mismo periodo, que el modelo no usó ni para entrenar ni para calibrar. Recall {p(mo["recall"])}: cumple.</li>
        <li><b>Derecha, datos futuros:</b> los meses siguientes (<code class="inl">fraudTest.csv</code>). Recall <b>{p(pm["recall"])}</b>: menos del 95 % prometido.</li>
        <li>¿Por qué? Los datos cambiaron: el fraude pasó de {p(fr["tasa_fraude"], 2)} a {p(fr["produccion_tasa_fraude"], 2)} y ya <b>no se parecen</b> a los de calibración.</li>
        <li><b>Lección:</b> hay que vigilar el recall y <b>recalibrar</b> con datos recientes.</li>
      </ul>
    </div>
  </div>'''))

S.append(slide("Riesgos de uso", "Cuándo la garantía deja de valer, o engaña", '''
  <table class="small">
    <tr><th style="width:27%">Riesgo</th><th style="width:38%">Qué pasa</th><th>Qué hacer</th></tr>
    <tr><td><b>Los datos cambian con el tiempo</b></td><td>La promesa se debilita (lo vimos: 92,7 % en vez de 95 %).</td><td>Vigilar y recalibrar con datos recientes</td></tr>
    <tr><td><b>La promesa es un promedio</b></td><td>Se cumple en total, pero puede fallar en un grupo.</td><td>CQR si la duda cambia; Mondrian si importa una clase</td></tr>
    <tr><td><b>Pocos datos de calibración</b></td><td>El resultado real oscila alrededor de lo prometido.</td><td>Idealmente 1.000 o más (por clase, en Mondrian)</td></tr>
    <tr><td><b>Usar calibración para entrenar</b></td><td>El modelo “ya vio” esos datos y la promesa se rompe.</td><td>Separar tres partes y usar la calibración una sola vez</td></tr>
    <tr><td><b>Explicarlo mal</b></td><td>“Este intervalo tiene 90 % de probabilidad” es incorrecto.</td><td>Decir: “este método acierta el 90 % de las veces”</td></tr>
  </table>'''))

S.append(slide("Riesgo · tamaño de calibración", "¿Cuántos datos de calibración necesito?", f'''
  <div class="grid2" style="grid-template-columns: 1.4fr 1fr">
    <div class="fig"><img src="../figuras/c_variabilidad_calibracion.png" alt="Variación de la cobertura según n de calibración"></div>
    <div>
      <p class="small" style="margin-bottom:12px">Si prometo 90 %, ¿qué obtengo en realidad? Depende de cuántos datos usé para calibrar:</p>
      <table class="small">
        <tr><th>Datos de calibración</th><th>Lo que realmente obtengo</th></tr>
        <tr><td class="num">50</td><td class="num">entre {p(var["50"]["p05"], 0)} y {p(var["50"]["p95"], 0)}</td></tr>
        <tr><td class="num">500</td><td class="num">entre {p(var["500"]["p05"], 0)} y {p(var["500"]["p95"], 0)}</td></tr>
        <tr><td class="num">5.000</td><td class="num">entre {p(var["5000"]["p05"], 0)} y {p(var["5000"]["p95"], 0)}</td></tr>
      </table>
      <p class="small muted" style="margin-top:12px">En el ejemplo, Mondrian calibró los fraudes con solo {fr["fraudes_cal"]} casos: por eso este número importa.</p>
    </div>
  </div>'''))

S.append(slide("Para sus proyectos", "Lo mismo con la librería MAPIE", f'''
  <div class="grid2" style="grid-template-columns: 1.6fr 1fr; gap:24px">
    <div>{code(MAPIE, 13.4)}</div>
    <div>
      <div class="card" style="margin-bottom:12px; padding:16px 20px"><h2>Siempre dos pasos</h2><p class="small"><code class="inl">conformalize</code> para calibrar y <code class="inl">predict_interval</code> o <code class="inl">predict_set</code> para predecir.</p></div>
      <div class="card" style="margin-bottom:12px; padding:16px 20px"><h2>Verificado</h2><p class="small">Con los XGBoost del ejemplo da <b>los mismos resultados</b> que el código hecho a mano.</p></div>
      <div class="card accent" style="padding:16px 20px"><h2>Mondrian</h2><p class="small">No viene incluido: se hace a mano como en la diapositiva 16.</p></div>
    </div>
  </div>'''))

S.append(slide("Aplicaciones reales", "¿Dónde se usa?", '''
  <div class="grid3">
    <div class="card"><h2>Descubrimiento de fármacos</h2><p>AstraZeneca lo usa para saber en cuáles predicciones sobre moléculas puede confiar (Norinder et al., 2014).</p></div>
    <div class="card"><h2>Pronóstico de ventas y energía</h2><p>Las empresas pronostican la demanda como un rango y no como un solo número. Librerías como MAPIE ya lo incluyen.</p></div>
    <div class="card"><h2>Medicina</h2><p>El modelo entrega una lista corta de diagnósticos posibles, con garantía, y el especialista toma la decisión final.</p></div>
    <div class="card"><h2>Riesgo y fraude</h2><p>Los bancos priorizan qué casos revisar y garantizan que detectan la mayoría de los fraudes, como en nuestro ejemplo.</p></div>
    <div class="card"><h2>Modelos de lenguaje</h2><p>Un chatbot entrega varias respuestas y se garantiza que al menos una es correcta (Quach et al., 2024).</p></div>
    <div class="card"><h2>Vehículos autónomos</h2><p>El vehículo calcula la zona donde puede estar un peatón, con garantía, y así planea una ruta segura.</p></div>
  </div>'''))

S.append(slide("Aplicación en mi trabajo · gastos administrativos en una aseguradora", "Clasificar nuevas combinaciones contables con garantía", '''
  <div class="grid2">
    <div>
      <p style="margin-bottom:14px">Bajo <b>NIIF 17</b>, cada combinación centro de costos–cuenta se clasifica como <b>atribuible</b> o <b>no atribuible</b> a los contratos de seguro, según la política corporativa.</p>
      <p>Un XGBoost aprende la clasificación vigente y <b>Mondrian</b> decide qué combinaciones nuevas se asignan solas, con garantía en <b>cada</b> categoría.</p>
    </div>
    <div>
      <div class="card" style="margin-bottom:14px"><p class="set" style="color:var(--blue)">{Atribuible} o {No atribuible}</p><p>Asignación automática y consistente con la política.</p></div>
      <div class="card"><p class="set" style="color:var(--violet)">{Atribuible, No atribuible}</p><p>Caso fronterizo: pasa a revisión del experto, con la sugerencia del modelo.</p></div>
    </div>
  </div>
  <div class="take">Y para el presupuesto: con <b>CQR</b>, entregar un rango confiable en lugar de un número único.</div>'''))

S.append(slide("Cierre", "Tres ideas para llevarse", '''
  <div style="display:flex; flex-direction:column; gap:22px; margin-top:8px">
    <p class="quote">1. Conformal le pone una garantía a <u>cualquier</u> modelo: solo mide sus errores en datos que no vio.</p>
    <p class="quote">2. Banda fija para empezar, CQR cuando la duda cambia y Mondrian cuando una clase importa más.</p>
    <p class="quote">3. La garantía vale mientras los datos no cambien: en producción, vigilar y recalibrar.</p>
  </div>'''))

S.append(slide("Bibliografía", "Referencias", '''
  <ul class="small" style="font-size:16px; columns:2; column-gap:40px">
    <li>Angelopoulos, A. N. &amp; Bates, S. (2023). Conformal prediction: A gentle introduction. <i>Foundations and Trends in ML</i>, 16(4).</li>
    <li>Chen, T. &amp; Guestrin, C. (2016). XGBoost: A scalable tree boosting system. <i>KDD</i>.</li>
    <li>Lei, J., G’Sell, M., Rinaldo, A., Tibshirani, R. J. &amp; Wasserman, L. (2018). Distribution-free predictive inference for regression. <i>JASA</i>, 113(523).</li>
    <li>Norinder, U. et al. (2014). Introducing conformal prediction in predictive modeling. <i>J. Chem. Inf. Model.</i>, 54(6).</li>
    <li>Papadopoulos, H., Proedrou, K., Vovk, V. &amp; Gammerman, A. (2002). Inductive confidence machines for regression. <i>ECML</i>.</li>
    <li>Quach, V. et al. (2024). Conformal language modeling. <i>ICLR</i>.</li>
    <li>Romano, Y., Patterson, E. &amp; Candès, E. J. (2019). Conformalized quantile regression. <i>NeurIPS</i>.</li>
    <li>Vovk, V. (2012). Conditional validity of inductive conformal predictors. <i>ACML</i>.</li>
    <li>Vovk, V., Gammerman, A. &amp; Shafer, G. (2005; 2.ª ed. 2022). <i>Algorithmic Learning in a Random World</i>. Springer.</li>
    <li>MAPIE: Model Agnostic Prediction Interval Estimator (v1.5). github.com/scikit-learn-contrib/MAPIE</li>
  </ul>'''))

S.append('''
<section class="slide cover">
  <h1>¿Preguntas?</h1>
  <div class="sub">Código reproducible: github.com/PedroV27/Conformal-Predictions</div>
  <div class="meta">Dataset: Credit Card Transactions Fraud Detection (Kaggle, kartik2112)<br>Python · NumPy · scikit-learn · XGBoost · MAPIE · matplotlib</div>
  <div class="foot"><span>Conformal Prediction</span><span class="pg"></span></div>
</section>''')

path.write_text(head + "\n" + "\n".join(S) + "\n\n</body>\n</html>\n", encoding="utf-8")
print("diapositivas:", len(S))
