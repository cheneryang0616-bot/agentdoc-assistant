# JVM 概念

## 定义
JVM（Java Virtual Machine，Java 虚拟机）是 Java 程序的运行环境，负责把 Java 字节码解释或即时编译为机器码并执行。它屏蔽了底层操作系统与硬件的差异，是 Java "一次编写、到处运行" 的基石。

## 核心构成
JVM 由类加载子系统、运行时数据区、执行引擎和本地方法接口组成。运行时数据区又分为堆、方法区、虚拟机栈、本地方法栈和程序计数器。

## 与 JRE / JDK 的关系
JDK 包含 JRE，JRE 包含 JVM。JVM 只负责执行字节码，不包含编译器 javac（javac 属于 JDK）。
