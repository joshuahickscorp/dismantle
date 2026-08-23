/*
 * DYLD interpose logger for G011 (NOETIC_ZERO_PARENT_RUNTIME_DEPENDENCY).
 *
 * Records every open / openat the process issues. Real I/O goes through
 * syscall(2) so the hook cannot recurse. Loaded via DYLD_INSERT_LIBRARIES;
 * log path is NOETIC_OPEN_LOG.
 *
 * macOS libc open is what Rust std::fs and CPython's builtin open call.
 * System binaries (SIP) ignore DYLD_INSERT; the adhoc greedy decode
 * binary and ~/.grok-vision/bin/python do not.
 */
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>

static int g_fd = -1;
static int g_busy = 0;

static void emit(const char *op, const char *path) {
    char buf[4096];
    int n;

    if (g_busy || g_fd < 0 || path == NULL || path[0] == '\0') {
        return;
    }
    n = snprintf(buf, sizeof buf, "%s\t%s\n", op, path);
    if (n <= 0) {
        return;
    }
    if (n >= (int)sizeof buf) {
        n = (int)sizeof buf - 1;
        buf[n] = '\n';
    }
    g_busy = 1;
    (void)!syscall(SYS_write, g_fd, buf, (size_t)n);
    g_busy = 0;
}

static void init(void) __attribute__((constructor));
static void init(void) {
    const char *p = getenv("NOETIC_OPEN_LOG");
    const char banner[] = "INTERPOSE_CTOR\n";
    (void)!write(STDERR_FILENO, banner, sizeof banner - 1);
    if (p == NULL || p[0] == '\0') {
        return;
    }
    g_fd = (int)syscall(SYS_open, p, O_WRONLY | O_CREAT | O_APPEND, 0644);
}

int my_open(const char *path, int flags, ...) {
    int mode = 0;
    emit("open", path);
    if (flags & O_CREAT) {
        va_list ap;
        va_start(ap, flags);
        mode = va_arg(ap, int);
        va_end(ap);
    }
    return (int)syscall(SYS_open, path, flags, mode);
}

int my_openat(int dirfd, const char *path, int flags, ...) {
    int mode = 0;
    emit("openat", path);
    if (flags & O_CREAT) {
        va_list ap;
        va_start(ap, flags);
        mode = va_arg(ap, int);
        va_end(ap);
    }
    return (int)syscall(SYS_openat, dirfd, path, flags, mode);
}

#define DYLD_INTERPOSE(_r, _o)                                                 \
    __attribute__((used)) static struct {                                      \
        const void *replacement;                                               \
        const void *replacee;                                                  \
    } _interpose_##_o __attribute__((section("__DATA,__interpose"))) = {       \
        (const void *)(unsigned long)(_r), (const void *)(unsigned long)(_o)}

DYLD_INTERPOSE(my_open, open);
DYLD_INTERPOSE(my_openat, openat);
