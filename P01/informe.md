# Simulador de sesgos conductuales: efecto disposición y exceso de confianza

**Proyecto 1 — Decision-Making Foundations & Behavioral Finance (ITESO), Prof. Luis Felipe Gómez Estrada.**
Pista principal: **A (efecto disposición)**; la pista B (exceso de confianza) se ejecuta completa como estimador de control cruzado.

| Rol | Responsable de |
|---|---|
| **[SIM] Arquitecto de simulación** | proceso generador de datos: precios, agentes, costos, mecanismos inyectados, reproducibilidad y la restricción de no-información |
| **[ECON] Econometrista conductual** | contabilidad PGR/PLR y sus convenciones, bootstrap por cuenta, regresiones de rotación brutas y netas, errores estándar, diagnóstico de la pendiente bruta |
| **[REF] Árbitro / editor científico** | pre-análisis, interpretación de cada escenario, límites de los estimadores y conclusiones |

**Fecha:** 10 de septiembre de 2026 · **Semilla maestra:** `20260910` · **Tiempo de ejecución:** ≈ 7.7 min en la última ejecución completa (la cifra exacta de esta ejecución está en la última celda).

**Cómo correrlo.** El simulador es `simulador.py`: `python simulador.py` corre los ocho escenarios con sus estimadores y guarda la tabla de resultados en `resultados/tabla_simulador.csv` (≈ 1 min). El informe completo es este cuaderno: se abre en Jupyter desde `C:\ifi\5\Comportamiento\proyecto_1` y se ejecuta con *Restart & Run All*. Reescribe `simulador.py` y los tres módulos que importa (`simlib_precios.py`, `simlib_agentes.py`, `simlib_estimadores.py`), las figuras (`figuras/`) y las tablas (`resultados/`), y exporta `informe.md`. Requiere Python ≥ 3.10 con numpy, pandas, scipy y matplotlib; las versiones usadas se imprimen abajo y en la última celda.

## 1. Resumen ejecutivo
**[REF]**

**Qué se inyectó.** Dos sesgos con magnitudes registradas antes de ver resultados:
* disposición $\delta$: el riesgo diario de venta se multiplica por $1\pm\delta$ según el signo frente al costo;
* sobre-precisión $\kappa$: escala el riesgo base.

Se aplicaron a 1,000 cuentas por escenario, sobre 100 valores y 756 días de un modelo de factores con alfa cero, y se recuperaron con los estimadores de Odean (1998) y Barber–Odean (2000).

**Qué se recuperó.**
* PGR − PLR crece con $\delta$ (E2 0.0276, E3 0.0846), y la razón de E2 (1.76) coincide con $(1+\delta)/(1-\delta)=1.86$.
* El nulo queda quieto (E1: 0.0007).
* La pendiente neta mide el costo por unidad de rotación (≈ −1.17 pp).
* La prueba de fuga es nula (p = 0.56).

**Qué se rompió.**
1. Rebalanceo y creencia en reversión, ambos con $\delta=0$, producen una disposición "significativa" (E7 0.0314, E8 0.0236). En E7, contar un recorte como una realización completa fabrica casi todo el efecto: con la convención fraccional queda en 0.0067.
2. Con disposición, la MCO de rotación da una β neta de **+4.30** ("operar paga") por causalidad inversa. La IV con la intensidad *ex ante* la corrige (-0.10).
3. PGR − PLR depende de la historia del mercado. Con $\delta=0.8$, su dispersión entre 20 mundos es 18.67 veces el error estándar del bootstrap por cuenta, que sólo vale condicional a la trayectoria.
4. La β neta no mide la dosis de $\kappa$: su pendiente de costos queda plana entre E4 y E5.
5. $\kappa$ infla PGR − PLR sin tocar $\delta$ (E6).

En el nulo, la β bruta rechaza al 5 %. El Monte Carlo lo identifica como un error tipo I (0 rechazos en 20 mundos), y se reporta sin cambiar la semilla. Se acertaron 33 de 38 predicciones registradas.

## 3. Diseño del simulador
**[SIM]**

**En una línea.** Una trayectoria de precios de un modelo de factores con alfa cero se genera **antes** que los agentes. Sobre ella, 1,000 cuentas por escenario operan con un riesgo diario de venta que se multiplica por $1+\delta$ sobre el costo promedio y por $1-\delta$ debajo de él; $\kappa$ escala el riesgo base. Recompran valores elegidos al azar el mismo día y pagan comisión y diferencial. El econometrista recibe sólo el "estado de cuenta": fotografías previas a la operación en cada día de venta, operaciones y valores diarios. El diseño está implementado en `simulador.py`, el archivo principal, que importa `simlib_precios.py` (precios), `simlib_agentes.py` (agentes y mecanismo inyectado) y `simlib_estimadores.py` (estimadores).

## 4. PRE-ANÁLISIS
La celda siguiente se registró (hash SHA-256 y hora UTC) antes de escribir el módulo de estimadores. Se reproduce sin cambios y la celda de código que la sigue verifica que coincide con el registro. El mismo párrafo está copiado, como comentarios (`#`) en texto plano, al inicio de cada archivo `.py` del proyecto.

**[REF] — PRE-ANÁLISIS (registrado antes de ejecutar cualquier estimador; no editado después)**

Inyectamos, con $N=1{,}000$ cuentas por escenario, 100 valores y 756 días, $\delta$ y $\kappa$ como **constantes poblacionales** ($\delta\in\{0,\,0.3,\,0.8\}$, $\kappa\in\{0,\,0.3,\,0.8\}$) sobre un riesgo base diario $h_0=\ell_i(1+3\kappa)/252$ con $\ell_i\sim U(0.25,1.25)$ y multiplicadores $1\pm\delta$ según el dominio respecto del costo promedio; los confusores usan $\delta=\kappa=0$ (rebalanceo: banda de 25 % sobre el peso igual, recorte al objetivo y compra de la posición más infraponderada; reversión: riesgo $\times 5$ si el rendimiento de 40 días supera 12 %). Esperamos, por la aritmética del riesgo condicionado a días de venta, $\widehat{PGR}\approx\widehat{PLR}\approx E[n]/E[n^2]\approx 0.05$ en el nulo y $PGR/PLR\approx(1+\delta)/(1-\delta)$ corregido a la baja por la heterogeneidad de composición: **E1** diferencia con IC que contiene 0 y $\beta_{bruta}$ (MCO e IV) con IC que contiene 0; **E2** diferencia $\approx 0.03$ (0.02–0.04), razón 1.5–2.2; **E3** diferencia $\approx 0.10$ (0.07–0.13), razón 5–11, y **misbehavior esperado**: $\beta_{bruta}^{MCO}>0$ de varios puntos por causalidad inversa, suficiente para que $\beta_{neta}^{MCO}$ no sea negativa, mientras la IV bruta queda en 0; **E4/E5** diferencia en 0 y $\beta_{neta}\approx -1.2$ pp por unidad de rotación anual (rango $-0.7$ a $-1.8$) **igual en ambos**, de modo que anticipamos que el requisito de monotonía de $|\beta_{neta}|$ de 4 a 5 **fallará**: la pendiente mide costo por unidad de rotación, no la dosis de $\kappa$; lo que sí crece es el lastre medio de costos (≈0.8, 1.5 y 2.6 %/año en E1, E4, E5); **E6** diferencia menor que en E3 (carteras más jóvenes, menos acumulación de perdedoras) y razón mayor o igual que en E3, con $\beta_{bruta}^{MCO}>0$ pero diluida respecto de E3; **E7** disposición espuria positiva y significativa (razón 2–5 contando cada recorte como una realización, y al menos 40 % menor con la convención fraccional), $\beta_{bruta}^{MCO}>0$ pequeña; **E8** disposición espuria positiva (razón 1.3–3, diferencia 0.015–0.05) y $\beta_{bruta}^{MCO}>0$. En todos los escenarios esperamos $\beta_{neta}-\beta_{bruta}=-\beta_{costos}$ exacta con rendimientos aritméticos; marcar a precio de ejecución desplaza $\beta_{bruta}$ en exactamente $-0.20$ pp; una ventana de caja de 21 días la desplaza $\approx-(21/252)\times 2.1\,\%\approx-0.2$ pp (el mercado realizado de la trayectoria, conocido porque los precios se generan antes que los agentes, rindió solo 2.1 %/año), probablemente no significativo; la regla de empates afectará menos de 0.1 % de las observaciones; FIFO y lote-por-lote atenuarán la diferencia frente al costo promedio (la referencia del propio agente); el SE por cuenta será ≈ igual al ingenuo en el nulo y mayor cuando $\delta>0$ o hay confusor; la prueba de fuga dará coeficientes nulos (p > 0.05 a 20 y 120 días) y la diferencia comprado-menos-vendido a 252 días será ≈ 0 frente a los −3.2 pp de Odean (1999); en la población heterogénea, corr($\delta,\kappa$) con IC que contiene 0 y corr($\delta$, rotación) negativa; con corr($\delta,\kappa$)=0.6 la $\beta_{bruta}^{MCO}$ sube mientras la IV con $\kappa$ sigue ≈ 0; la curva de recuperación será creciente y convexa en $\delta$, con razón indefinida en $\delta=1$ ($PLR=0$); y la dispersión de $\beta_{bruta}$ entre trayectorias de precios independientes superará el SE bootstrap por cuenta (factor 1.2–3) porque las cuentas comparten valores.

*Antes de este registro solo se ejecutó una prueba mecánica del simulador (identidad contable, tiempo de cómputo y rotación media como calibración del DGP); ningún estimador de PGR/PLR ni de $\beta$ estaba implementado. Nada de lo que aparece debajo de esta celda se usó para revisarla.*

### 7.2 Tabla maestra de los ocho escenarios
**[ECON]** Diferencias y razones con SE por **bootstrap de cuentas** (1,000 réplicas, IC percentil 95 %). Pendientes en **puntos porcentuales de rendimiento anual por unidad de rotación anual** (1 = 100 % de la cartera al año), con SE bootstrap por cuenta y HC1. IV = 2SLS con la intensidad *ex ante* $h_{0,i}$ como instrumento de la rotación realizada. La columna *Pre-análisis* cuenta las predicciones codificadas que se cumplieron.

| Esc. | δ | κ | Confusor | PGR | PLR | PGR−PLR (EE) | IC 95 % | PGR/PLR (EE) | β bruta (EE boot / HC1) | β neta (EE boot / HC1) | β IV bruta (EE) | β IV neta (EE) | Rotación media | N cuentas | N obs. | Pre-análisis |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | 0.0 | 0.0 | — | 0.0506 | 0.0498 | 0.0007 (0.0005) | [-0.0002, 0.0017] | 1.01 (0.01) | +1.39 (0.61 / 0.61) | +0.17 (0.63 / 0.62) | 0.83 (0.63) | -0.50 (0.65) | 0.67 | 1000 | 793,855 | 2/3 |
| S2 | 0.3 | 0.0 | — | 0.0641 | 0.0365 | 0.0276 (0.0006) | [0.0265, 0.0289] | 1.76 (0.02) | +1.71 (0.55 / 0.58) | +0.68 (0.56 / 0.59) | 0.29 (0.65) | -0.92 (0.66) | 0.67 | 1000 | 780,830 | 5/5 |
| S3 | 0.8 | 0.0 | — | 0.0973 | 0.0127 | 0.0846 (0.0010) | [0.0827, 0.0868] | 7.66 (0.12) | +5.26 (0.58 / 0.61) | +4.30 (0.59 / 0.61) | 1.05 (0.70) | -0.10 (0.70) | 0.63 | 1000 | 656,085 | 7/7 |
| S4 | 0.0 | 0.3 | — | 0.0515 | 0.0522 | -0.0008 (0.0004) | [-0.0015, -0.0000] | 0.98 (0.01) | +0.33 (0.34 / 0.34) | -0.84 (0.35 / 0.35) | 0.21 (0.35) | -1.07 (0.37) | 1.24 | 1000 | 1,444,318 | 3/4 |
| S5 | 0.0 | 0.8 | — | 0.0552 | 0.0552 | 0.0000 (0.0003) | [-0.0006, 0.0006] | 1.00 (0.01) | +0.09 (0.19 / 0.20) | -1.06 (0.22 / 0.22) | 0.06 (0.19) | -1.17 (0.22) | 2.19 | 1000 | 2,353,100 | 5/5 |
| S6 | 0.8 | 0.8 | — | 0.1157 | 0.0157 | 0.1000 (0.0010) | [0.0981, 0.1022] | 7.38 (0.06) | +1.74 (0.28 / 0.28) | +0.69 (0.30 / 0.29) | 0.03 (0.32) | -1.13 (0.34) | 1.76 | 1000 | 1,837,305 | 4/6 |
| S7 | 0.0 | 0.0 | rebalanceo | 0.0663 | 0.0349 | 0.0314 (0.0005) | [0.0306, 0.0324] | 1.90 (0.01) | +0.69 (0.46 / 0.48) | -0.47 (0.48 / 0.49) | 0.55 (0.50) | -0.76 (0.52) | 0.97 | 1000 | 1,846,218 | 3/4 |
| S8 | 0.0 | 0.0 | reversión | 0.0647 | 0.0412 | 0.0236 (0.0005) | [0.0226, 0.0245] | 1.57 (0.01) | +0.76 (0.38 / 0.37) | -0.22 (0.39 / 0.38) | -0.29 (0.39) | -1.35 (0.40) | 1.14 | 1000 | 1,222,056 | 4/4 |

**[REF] — Interpretación de la validación (§7.5)**

1. **Nulo.**
   - La disposición queda quieta en E1 (IC [-0.0002, 0.0017]). La β bruta MCO, +1.39, **no**.
   - Antes de interpretar nada se depuró. En 20 mundos independientes, con mercado y población nuevos, la β bruta de E1 promedia +0.01. Su DE (0.59) coincide con el EE bootstrap medio (0.60; cociente 0.98), y la prueba al 5 % rechazó en el 0 % de los mundos. El estimador es insesgado y está bien calibrado: el rechazo de esta semilla es una cola de una distribución correcta y se reporta sin cambiar la semilla.
   - E4 es el espejo del lado de la disposición: -0.0008, con IC [-0.0015, -0.0000], justo en el borde. Con unas quince pruebas nulas en la tabla, dos rechazos marginales al 5 % están dentro de lo esperable.
2. **Monotonía.**
   - PGR − PLR crece de E2 a E3 sin traslape de los IC.
   - $|\beta_{neta}|$ crece de E4 a E5 (β neta -0.84 → -1.06) sólo en la estimación puntual, y no por la razón que supone el enunciado: la pendiente de costos casi no cambia (1.17 → 1.16).
   - En el barrido de $\kappa$, $|\beta_{neta}|$ toma los valores 1.32, 0.48, 0.48, 0.70, 1.18, 1.17: no es monótona. El costo medio, en cambio, sube de 0.74 a 3.11 pp/año. La dosis de $\kappa$ está en el nivel, no en la pendiente.
3. **Silencio cruzado.** La disposición calla cuando sólo actúa $\kappa$. La confianza **no calla** cuando sólo actúa $\delta$: la MCO bruta de E3 es +5.26 por causalidad inversa (§9), mientras la IV queda en +1.05.
4. **Curva de recuperación.**
   - Para $\delta=0,0.2,\dots,1$, PGR − PLR vale 0.0005, 0.0192, 0.0383, 0.0598, 0.0843, 0.1150. Es creciente y convexa (incrementos de 0.0186, 0.0191, 0.0215, 0.0246, 0.0307), porque las perdedoras se acumulan (las ganancias en papel bajan de 50.5 % a 42.4 %) y reducen el denominador de los días de venta.
   - La aproximación lineal de primer orden sólo vale cerca de $\delta=0$, igual que la de Arrow–Pratt en `001-vnm.tex`.
   - La razón (1.01, 1.47, 2.18, 3.58, 7.54) se aleja cada vez más por debajo de $(1+\delta)/(1-\delta)$ y en $\delta=1$ no existe ($PLR=0$). Es inestable justo donde la disposición es más fuerte.
5. **Lo que nadie pidió y más importa.**
   - En E3, la DE de PGR − PLR entre mundos es 0.0194 (rango 0.0619–0.1333), **18.67 veces** el EE bootstrap por cuenta (0.0010), y su correlación con el rendimiento del mercado de cada mundo es -0.80: en mercados bajistas se acumulan perdedoras y la diferencia crece.
   - El bootstrap por cuenta es correcto *condicional a la historia de precios*. Como estimador de $\delta$, un solo historial, como el de Odean, informa bastante menos de lo que sugiere su IC. Para β en E3 el cociente es 2.55: la causalidad inversa también depende de la trayectoria.
   - La trayectoria de la tabla maestra tiene además su propia huella. Su reversión transversal realizada (`realized_reversal`, la correlación entre el rendimiento residual de 126 días pasados y el de los 126 siguientes; negativa = reversión) es -0.058, contra una media de +0.024 entre mundos (DE 0.051; z = -1.6). Entre mundos, la β bruta se correlaciona con ella en -0.19 en E1 y -0.24 en E5: más reversión, más β bruta. Seis poblaciones heterogéneas nuevas sobre esta misma trayectoria dan una IV(h₀) bruta media de +0.46 (DE 0.24): esta historia premió un poco la rotación sin que nadie tuviera información.

![Figura (b): recuperación de δ y barrido de κ](figuras/fig_b_recuperacion.png)

![Figura (c): pendientes bruta y neta por escenario](figuras/fig_c_pendientes.png)

## 11. Lo que los estimadores no pueden distinguir
**[REF]**

**PGR − PLR.** Los cuatro conteos responden una sola pregunta: ¿se realiza con más frecuencia lo que está por encima del costo? En esta simulación, tres mecanismos distintos le dan la misma respuesta. Disposición genuina (E3: 0.0846), rebalanceo con $\delta=0$ (E7: 0.0314) y creencia falsa en la reversión con $\delta=0$ (E8: 0.0236) salen los tres positivos y muy significativos. Con datos reales se agregan dos rivales más que dejan la misma huella. Para cada uno, estos son los datos que rompen el empate:

| Rival de la disposición genuina | Qué lo delata | Evidencia en esta simulación |
|---|---|---|
| **Rebalanceo mecánico** | Pesos objetivo o mandato de la cuenta; datos a nivel de orden que separen salidas totales de recortes; destino de lo obtenido (el rebalanceador compra las infraponderadas, es decir las perdedoras). | En E7, el 75 % de las ganancias realizadas son recortes (E3: 25 %). Con la convención fraccional la diferencia cae a 0.0067. La tasa de realización de una ganancia sobreponderada es 0.995 contra 0.023 si no lo está; en E3 el peso casi no importa (0.100 contra 0.096). |
| **Creencia en reversión** | Variación transversal de la señal de reversión (el movimiento reciente) que un rebalanceador ignoraría y un creyente no; tasas de venta condicionadas a la vez al signo frente al costo y al rendimiento reciente (riesgo de venta al estilo Grinblatt–Han, en vez de un cociente de conteos). | En E8, una **pérdida** que subió más de 12 % en 40 días se realiza con tasa 0.128 y una ganancia que no subió con 0.032: manda el movimiento reciente, no el precio de compra. En E3 es al revés: 0.100 contra 0.012. |
| **Motivo fiscal** | Momento de las operaciones alrededor del cierre fiscal. La venta fiscal realiza pérdidas y se concentra en diciembre, así que con datos reales el efecto se atenúa o se invierte ahí. | Este simulador no tiene calendario ni impuestos, así que no puede probarlo (§13). |
| **Necesidad de liquidez correlacionada con rendimientos pasados** | Retiros de efectivo y comportamiento de reinversión después de la venta: quien vende por liquidez retira lo obtenido y quien vende por disposición lo reinvierte. También puntos de referencia declarados en encuestas. | En el simulador toda venta se reinvierte. Su intensidad base $\ell_i$ es liquidez pura y no depende del signo, de ahí que E1 salga nulo. |

A eso se suma un problema de medición que los cuatro conteos también esconden. PGR y PLR se miden sólo en días de venta, así que mezclan el riesgo de venta con la composición de la cartera y con cuántas ventas caen el mismo día. En E6 la diferencia aumentó con $\kappa$ aunque el cociente de riesgos inyectado seguía siendo $1.8/0.2=9$. Un estimador de **riesgo de venta** (ventas por posición-día en ganancia contra en pérdida, sobre *todos* los días) recuperaría $(1+\delta)/(1-\delta)$ sin depender de $\kappa$ ni de $n_i$. Para eso hacen falta las tenencias diarias completas y no sólo los días de venta.

**La pendiente de Barber–Odean.** En la sección transversal se parecen tres historias: el lastre de costos, la causalidad inversa y la desinformación genuina (lo comprado rinde menos que lo vendido, los −3.2 pp de Odean 1999). En E3 la MCO neta da **+4.30**: con datos reales se leería "operar paga", y lo que hay es causalidad inversa pura (IV neta -0.10). Los datos que desempatan son tres:
1. **Una medida ex ante de la rotación que se pretendía**, que el rendimiento no pueda causar: aquí $h_{0,i}$; con datos reales, el género (Barber–Odean 2001), la rotación de un periodo previo o el tipo de cuenta.
2. **Variación de costos dentro de la misma cuenta**, como un cambio de tarifa: el lastre de costos responde a ella y la desinformación no.
3. **El rendimiento futuro de lo comprado menos lo vendido** (Odean 1999): la desinformación lo vuelve negativo antes de costos y el lastre de costos no. Aquí da +0.46 pp (EE 0.33).

Además, la pendiente neta mide **costo por unidad de rotación** (−β costos ≈ −1.17 en E4 y −1.16 en E5) y **no la dosis de exceso de confianza**. La dosis se ve en el nivel: rotación media y costo medio por grupo, que es justo lo que reportan los quintiles de Barber–Odean (2000).

## 12. Conclusiones
**[REF]** Sobre los **estimadores**, no sobre los inversores.

1. **PGR − PLR recupera el signo y el orden de $\delta$, pero no una magnitud que se pueda llevar a otra población.**
   - La relación con $\delta$ es convexa, no lineal.
   - Depende de $n_i$: el nivel del nulo es $E[n]/E[n^2]$, y por eso nuestro PGR nulo (0.0506) es un tercio del de Odean.
   - Depende de $\kappa$: en E6 sube 18.2 % sin que cambie $\delta$.
   - Depende de la historia del mercado: su DE entre mundos es 18.67 veces el EE.
   - La razón viaja mejor, pero explota cuando $\delta\to1$.
2. **Falsos positivos que habrían pasado con datos reales.**
   - E7 (rebalanceo) y E8 (reversión) aparecen como disposición, con $\delta=0$ y p < 0.001.
   - E3 produce además un falso "operar paga" (β neta MCO +4.30).
   - E1 y E4 dan rechazos marginales de los nulos (β bruta y diferencia). El Monte Carlo sobre E1 y E5 muestra estimadores insesgados y bien calibrados, así que se tratan como errores tipo I.
3. **La pendiente de Barber–Odean recupera el costo, no $\kappa$.**
   - −β costos está entre 1.00 y 1.26 pp por unidad de rotación en todo el barrido de $\kappa$.
   - La identidad bruto/neto es exacta.
   - $\kappa$ sólo se ve en los niveles: rotación y costo medio.
   - En la línea base ninguna fuente mecánica produce una pendiente bruta negativa (§9): la caja de 21 días la mueve -0.68 pp, el diferencial dentro del precio -0.199 (el costo de ida y vuelta) y la capitalización menos de 0.05. La fuente dominante es la causalidad inversa, que la empuja hacia arriba.
4. **El SE correcto depende de la pregunta.**
   - Condicional a la trayectoria, el bootstrap por cuenta es el SE correcto, y el ingenuo lo subestima hasta 2.60 veces (en el nulo no lo subestima: cociente 1.02, como se registró).
   - Para la pregunta poblacional —cuánto vale $\delta$—, la dispersión entre historias de mercado domina.
5. **La IV con una intensidad *ex ante* elimina la causalidad inversa** (E3: +5.26 → +1.05). No elimina la huella de la trayectoria: sobre esta semilla, las réplicas dan una IV bruta ≈ +0.46.

Se acertaron 33 de 38 predicciones registradas. Las cinco fallas —β bruta de E1, diferencia de E4, las dos de E6 y el rango de la razón en E7— están documentadas arriba, y ninguna se corrigió cambiando semillas ni parámetros.
