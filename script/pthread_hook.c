#define _GNU_SOURCE
#include <pthread.h>
#include <stdio.h>
#include <dlfcn.h>
#include <stdlib.h>

// Define a pointer to the original pthread_create function
static int (*original_pthread_create)(pthread_t *, const pthread_attr_t *, void *(*)(void *), void *) = NULL;

// Hooked pthread_create function
int pthread_create(pthread_t *thread, const pthread_attr_t *attr, void *(*start_routine)(void *), void *arg) {
    // Load the original pthread_create function
    if (!original_pthread_create) {
        original_pthread_create = dlsym(RTLD_NEXT, "pthread_create");
        if (!original_pthread_create) {
            fprintf(stderr, "Error: unable to find original pthread_create\n");
            exit(EXIT_FAILURE);
        }
    }

    // Print debug information
    printf("[HOOK] pthread_create called\n");

    // Call the original pthread_create function
    int result = original_pthread_create(thread, attr, start_routine, arg);

    // Print the thread ID if the thread was successfully created
    if (result == 0) {
        printf("[HOOK] Thread created: %lu\n", *thread);
    } else {
        fprintf(stderr, "[HOOK] pthread_create failed with error: %d\n", result);
    }

    return result;
}

// Hook other functions as needed (e.g., pthread_join, pthread_exit)
