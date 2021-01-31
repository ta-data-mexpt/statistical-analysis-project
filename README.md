# Recomendador de libros en Goodreads

En este proyecto elaboré un recomendador de libros. El recomendador utiliza la información del perfil de un usuario, así como de sus amigos y personas a las que sigue, para sugerirle al usuario cuál es próximo libro que debe leer. Para recopilar la información que alimenta al recomendador, utilicé la API de Goodreads. 

#### Algunos aspectos que me gustaría incluir después:
- Incluir en el análisis los usuarios con forbidden access.
- Extraer automáticamente la lista de amigos de un usuario y que se incluyan en el análisis.

#### Mejoras pendientes a mi código:
- Hay libros cuyo título está en inglés y que otros leyeron en español.
- La fórmula para recomendar libros puede que no esté calibrada correctamente. Temo que le esté dando mucho peso a ciertos libros y muy poco peso a libros de las personas con las que tengo correlación negativa. 
- Quizá mi fórmula de weighted_rating no es del todo correcta.