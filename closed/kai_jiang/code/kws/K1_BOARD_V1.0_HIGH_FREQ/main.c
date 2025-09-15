
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/select.h>
#include "uart.h"
#include "internally_implemented.h"
#include "submitter_implemented.h"

int uart_fd = -1;

void handle_data(int fd) {
    char buffer[256];
    ssize_t bytes_read = read(fd, buffer, sizeof(buffer) - 1);
    if (bytes_read > 0) {
        // buffer[bytes_read] = '\0';
        // th_printf("Received: %s\n", buffer);
        for(int i = 0; i < bytes_read; i++)
		{
			ee_serial_callback(buffer[i]);
		}

    } else if (bytes_read == -1) {
        perror("read");
    }
}

int main() {
    uart_fd = open_serial("/dev/ttyS7", B115200);
    if (uart_fd == -1) {
		printf("uart open error!\n");
        return -1;
    }

	ee_benchmark_initialize();

	fd_set read_fds;
    int max_fd;

    while (1) {
        FD_ZERO(&read_fds);
        FD_SET(uart_fd, &read_fds);
        max_fd = uart_fd + 1;

        // 使用 select 监听文件描述符
        int activity = select(max_fd, &read_fds, NULL, NULL, NULL);

        if (activity == -1) {
            perror("select");
            close(uart_fd);
            return -1;
        }

        if (FD_ISSET(uart_fd, &read_fds)) {
            handle_data(uart_fd);
        }
    }

    close(uart_fd);
    return 0;
}

