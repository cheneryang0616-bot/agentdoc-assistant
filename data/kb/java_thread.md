# Java 线程模型

## 线程与 JVM
Java 线程与操作系统原生线程一一映射，由 JVM 调度。线程共享堆内存，各自拥有独立的虚拟机栈与程序计数器。

## 并发工具
Java 提供 synchronized、volatile、以及 java.util.concurrent 包中的线程池、锁、并发容器等工具。

## 内存模型
Java 内存模型（JMM）定义了多线程下的可见性与有序性规则，happens-before 是其核心概念。
