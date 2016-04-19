### Компиляция и запуск
Compilation:

```{bash}
./install.sh
```

Execute the application:

```{bash}
./build/execute.sh
```
Запускается 5 узлов, последний из них -- серверный узел.

### Описание
`guiPlot` -- папка с отдельным проектом на Qt c++, который при компиляции создает динамическую библиотеку, использующуюся в основном проекте.

`LogSystem` -- папка с исходниками системы логирования.

### TODO
-  UML диаграмма программы
-  Разобраться как работать с [ffmpeg](https://trac.ffmpeg.org/wiki/Create%20a%20video%20slideshow%20from%20images).
