# Decisiones editoriales — Fase 3

## 1. No presentar el modelo como hidrodinámico
Se decidió corregir el lenguaje metodológico para evitar la impresión de que StoHyMoLAP resuelve ecuaciones hidrodinámicas espaciales. El texto ahora declara que el núcleo físico es lumped rainfall--runoff. Esta decisión protege el manuscrito frente a una crítica probable de reviewers de Journal of Hydrology.

## 2. Mantener el núcleo RAMIS, pero explicar sus límites
No se modificaron las ecuaciones centrales del núcleo RAMIS. Sin embargo, se añadió una explicación más clara sobre su interpretación como modelo lumped de descarga y memoria, no como representación espacial de procesos hidráulicos.

## 3. Eliminar placeholders visibles
El placeholder de área, coordenadas y periodo de registro se eliminó del texto visible. En su lugar se dejó una nota LaTeX oculta para el autor. Esto evita que el manuscrito parezca incompleto durante una lectura editorial, pero conserva la advertencia de que esos metadatos deben añadirse antes del envío.

## 4. No inventar metadatos de estación
No se inventaron coordenadas, área de drenaje, elevación ni periodo completo de registro de Puente Ramis. Esos datos deben insertarse desde fuentes oficiales SENAMHI/ANA.

## 5. Interpretar XGB como gradient boosting, no como paquete específico
Dado que el backend real es `sklearn_hgb`, los resultados etiquetados como XGB se interpretan como evidencia de gradient-boosted tree emulation, no como evidencia específica de la librería `xgboost`.

## 6. Interpretar GRU-labelled como fallback MLP
E7 y E8 se mantienen en la matriz experimental, pero se interpretan como experimentos con features secuenciales ejecutados mediante MLP sobre secuencias aplanadas. No deben usarse para afirmar superioridad de redes recurrentes.

## 7. Reducir citas literales largas
Se reemplazaron citas literales extensas por paráfrasis académicas. Esto mejora el tono editorial y reduce el riesgo de que el manuscrito parezca una revisión bibliográfica ensamblada por citas.

## 8. Separar función del ensemble físico
Se decidió distinguir dos usos del ensemble físico:
1. predictor físico estocástico directo;
2. fuente de descriptores cuantílicos para ML.

Esta separación es central para defender el aporte metodológico del manuscrito.

## 9. Mantener la auditoría como parte del método
La auditoría de leakage, backend y ventana común no se trató como material suplementario; se mantuvo dentro del método porque es uno de los aportes diferenciales del manuscrito.
