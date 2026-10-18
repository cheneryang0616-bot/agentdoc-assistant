# Python GIL

## 全局解释器锁
GIL（Global Interpreter Lock，全局解释器锁）是 CPython 的一项机制，保证同一时刻只有一个线程执行 Python 字节码。

## 影响
GIL 使得 CPU 密集型多线程无法真正并行，需改用多进程；但 I/O 密集型任务仍可受益于多线程，因为阻塞时会释放 GIL。

## 与 Java 对比
Java 的线程模型允许多线程在多个 CPU 核心上真正并行，这是 Python 与 Java 并发能力的重要差异。
