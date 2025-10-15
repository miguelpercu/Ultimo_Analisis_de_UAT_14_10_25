🌌 Resolución de la Tensión de Hubble a través del Marco UAT
Repositorio oficial para el manuscrito: Resolution of the Hubble Tension through the Unified Applicable Time (UAT) Framework: Quantum Gravitational Effects in the Early Universe

Este repositorio contiene todos los archivos de datos, código y documentos suplementarios necesarios para reproducir completamente el análisis presentado en el manuscrito. El Marco de Tiempo Unificado Aplicable (UAT), motivado por principios de Gravedad Cuántica de Lazos (LQG), ofrece una solución físicamente coherente y estadísticamente robusta a la Tensión del H 
0
​
 .

✨ Resultados Clave (Evidencia Decisiva)
Los resultados estadísticos centrales del análisis Bayesiano MCMC, combinando datos del Fondo Cósmico de Microondas (CMB), Oscilaciones Acústicas de Bariones (BAO) y Supernovas Ia (SNe Ia), demuestran el rendimiento superior del modelo UAT sobre ΛCDM.

Parámetro	Resultado Óptimo UAT	ΛCDM (Planck)	Conclusión Estadística
Constante de Hubble (H 
0
​
 )	73.02±0.82 km/s/Mpc	67.36 km/s/Mpc	Tensión Resuelta (Coincide con mediciones locales)
Mejora Estadística (Δχ 
2
 )	+40.389	0.00	Ajuste Superior sobre el modelo óptimo ΛCDM
Evidencia Bayesiana (lnB 
01
​
 )	12.64	0.00	Evidencia Decisiva (Umbral > 5)
Horizonte del Sonido (r 
d
​
 )	141.75±1.1 Mpc	147.1 Mpc	Reducción Requerida (∼3.6%) alcanzada

Exportar a Hojas de cálculo
📁 Estructura del Repositorio
Archivo	Descripción
Manuscrito_UAT_14_10_25 (1).pdf	Manuscrito Principal conteniendo el análisis y la discusión completos.
supplementary_*.pdf	Colección de Información Suplementaria detallando la fundación teórica, el parámetro k 
early
​
 , y el análisis de pulls con DESI BAO.
UAT_realistic_analysis_final_EN.py	Código de Reproducibilidad. Script Python con la definición del modelo UAT, cálculos χ 
2
  y lógica de optimización (traducido al inglés).
simulated_mcmc_chains (1).dat	Datos crudos de las Cadenas MCMC utilizadas para derivar los límites de los parámetros.
UAT_corner_plot (1).png	Visualización del corner plot de las distribuciones de probabilidad posteriores para los parámetros UAT.
Expansion_UAT_vs_LCDM.png	Gráfico comparando la tasa de expansión E(z) del UAT frente a ΛCDM.

Exportar a Hojas de cálculo
🛠️ Reproducibilidad y Uso del Código
El análisis puede ser reproducido utilizando el script Python provisto.

Requisitos
Python 3.x

Librerías científicas estándar: numpy, pandas, scipy, matplotlib

Instrucciones de Ejecución
Clonar el repositorio (o descargar los archivos):

Bash

git clone https://github.com/miguelpercu/Ultimo_Analisis_de_UAT_14_10_25
cd Ultimo_Analisis_de_UAT_14_10_25
Ejecutar el script principal de análisis:

Bash

python UAT_realistic_analysis_final_EN.py
El script generará las figuras, actualizará el archivo de resultados ejecutivos y mostrará las restricciones finales de los parámetros.

👨‍🔬 Autor y Contacto
Miguel Angel Percudani
Especialista en Radiación y Campos

Email: miguel_percudani@yahoo.com.ar
Ubicación: Puan, Buenos Aires, Argentina
Perfil GitHub: miguelpercu
