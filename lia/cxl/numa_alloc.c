#include <numa.h>
#include <numaif.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

void check_memory_node(void *memory, int num) {
    int status[num];    // Array to hold the status of each page
    void *pages[num];   // Array of pointers to each page in the allocated memory
    size_t page_size = getpagesize();
    for (size_t i = 0; i < num; i++) {
        pages[i] = (char *)memory + i * page_size;
    }

    if (numa_move_pages(0, num, pages, NULL, status, 0) != 0) {
        perror("Error checking NUMA node of memory pages");
        return;
    }

    for (size_t i = 0; i < num; i++) {
        printf("Page %zu is on node %d\n", i, status[i]);
    }
}

void* numa_alloc_node(size_t size, int node) {
    struct bitmask *mask = numa_bitmask_alloc(numa_max_node() + 1);

    // Check if NUMA is available
    if (numa_available() == -1) {
        fprintf(stderr, "NUMA is not available\n");
        numa_bitmask_free(mask);
        return NULL;
    }

    // Save previous interleave mask and strict mode
    struct bitmask *old_mask = numa_get_interleave_mask();

    // Set the interleave mask on the specified NUMA nodes
    numa_bitmask_setbit(mask, node);
    numa_set_interleave_mask(mask);

    // Set strict mode to bind memory to the specified node
    numa_set_strict(1);

    // Allocate memory on the specified NUMA node
    void *memory = numa_alloc(size);
    if (!memory) {
        fprintf(stderr, "Memory allocation failed on node 2,3\n");
        // Restore previous settings
        numa_set_interleave_mask(old_mask);
        numa_set_strict(0);
        numa_bitmask_free(old_mask);
        numa_bitmask_free(mask);
        return NULL;
    }

    // Restore previous settings
    numa_set_interleave_mask(old_mask);
    numa_set_strict(0);
    numa_bitmask_free(old_mask);
    numa_bitmask_free(mask);

    return memory;
}

void* numa_alloc_interleave(size_t size) {
    struct bitmask *mask = numa_bitmask_alloc(numa_max_node() + 1);

    // Check if NUMA is available
    if (numa_available() == -1) {
        fprintf(stderr, "NUMA is not available\n");
        numa_bitmask_free(mask);
        return NULL;
    }

    // Save previous interleave mask and strict mode
    struct bitmask *old_mask = numa_get_interleave_mask();

    // Set the interleave mask on the specified NUMA nodes
    numa_bitmask_setbit(mask, 2);
    numa_bitmask_setbit(mask, 3);
    numa_set_interleave_mask(mask);

    // Set strict mode to bind memory to the specified node
    numa_set_strict(1);

    // Allocate memory on the specified NUMA node
    void *memory = numa_alloc(size);
    if (!memory) {
        fprintf(stderr, "Memory allocation failed on node 2,3\n");
        // Restore previous settings
        numa_set_interleave_mask(old_mask);
        numa_set_strict(0);
        numa_bitmask_free(old_mask);
        numa_bitmask_free(mask);
        return NULL;
    }

    // Restore previous settings
    numa_set_interleave_mask(old_mask);
    numa_set_strict(0);
    numa_bitmask_free(old_mask);
    numa_bitmask_free(mask);

    return memory;
}

void numa_free_node(void *memory, size_t size) {
    numa_free(memory, size);
}