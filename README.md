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

### TODO
-  добавить систему [логирования](http://www.drdobbs.com/cpp/a-lightweight-logger-for-c/240147505?pgno=1).
-  UML диаграмма программы
-  Разобраться как работать с [ffmpeg](https://trac.ffmpeg.org/wiki/Create%20a%20video%20slideshow%20from%20images).