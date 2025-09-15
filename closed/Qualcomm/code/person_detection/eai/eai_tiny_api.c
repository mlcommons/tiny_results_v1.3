#include "eai_tiny_api.h"
#include "platform.h"
#include "eai_log.h"
#include <string.h>

#define ALIGN(n, alignment) ((n + alignment - 1) & (~(alignment - 1)))

static const char *data_type_string[] = {
    "",      "float32", "uint8", "int8",    "uint16",  "int16",  "int32",
    "int64", "",        "",      "float16", "float64", "uint32", "uint64",
};

// static const char* hw_cfg_heap[] = {
//     NULL,
//     "CAM_LLCC_ISLAND1_POOL",
//     "AUDIO_ISLAND_LPASS_TCM_POOL",
//     "AUDIO_ISLAND_TCM_PHYSPOOL"
// };

uint8_t* allocate_and_align_model_buffer(size_t model_size) {
    uint32_t model_buffer_alignment;
    EAI_RESULT eai_ret = eai_get_property(NULL, EAI_PROP_MODEL_ALIGNMENT, &model_buffer_alignment);
    LOG_AND_RETURN_IF_TRUE(eai_ret != EAI_SUCCESS, NULL);

    return (uint8_t *)malloc_align(model_buffer_alignment, model_size);
}

size_t get_file_size(const char* file_name) {
    FILE* fp = fopen(file_name, "rb");
    if (fp == NULL) {
        printf("Cannot open file: %s\n", file_name);
        fclose(fp);
        return 0;
    }

    size_t size = 0;
    fseek(fp, 0, SEEK_END);
    size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    fclose(fp);

    return size;
}

int read_file_from_handle(uint8_t* buffer, size_t size, FILE* fp)
{

    size_t read_size = fread(buffer, 1, size, fp);
    if (read_size == 0) {
        //End of the file
        return -2;
    }
    return 0;
}

int read_file(uint8_t* buffer, size_t size, const char * file_name)
{
    int ret = 0;
    FILE* fp = fopen(file_name, "rb");
    if (fp == NULL) {
        printf("Cannot open file: %s\n", file_name);
        ret = -1;
    } else {
        ret = read_file_from_handle(buffer, size, fp);
    }
    fclose(fp);

    return ret;
}

int load_model(struct eai_run_context *context) {
    printf("Model : %s\n", context->model_name);
    context->model_size = get_file_size(context->model_name);
    context->model_buffer = allocate_and_align_model_buffer(context->model_size);
    printf("Model size: %zu\n", context->model_size);
    if (read_file(context->model_buffer, context->model_size, context->model_name)) {
        return 1;
    }
    return 0;

}


uint32_t get_eai_init_flags() {
    uint32_t eai_init_flags = 0x0;
  
    return eai_init_flags;
}

EAI_RUN_RESULT set_configurations(struct eai_run_context * context){

    if (!context->model_buffer) {
        return EAI_RUN_EAI_FAILURE;
    }

    return EAI_RUN_SUCCESS;
}

EAI_RUN_RESULT init_eai(struct eai_run_context *context)
{
    EAI_RESULT eai_ret = EAI_SUCCESS;
    eai_memory_info_t scratch_memory;
    eai_memory_info_t persistent_memory;

    uint32_t eai_init_flags = get_eai_init_flags();
    EAI_LOG("eai_init_flags %lu\n", eai_init_flags);
    if (!context->model_buffer)
    {
        return EAI_RUN_ERROR;
    }
    eai_memory_info_t model_buffer = {0};
    model_buffer.addr = context->model_buffer;
    model_buffer.memory_size = context->model_size;
    model_buffer.memory_type = EAI_MEM_TYPE_DDR;

	// eai_ret = eai_init_ex(&context->eai_handle, context->model_buffer, context->model_size, eai_init_flags, hw_cfg_heap[context->hw_cfg_heap_type]); // flags contain user set flags such as enable/disable lpi mode
    eai_ret = eai_init(&context->eai_handle, &model_buffer, eai_init_flags); // flags contain user set flags such as enable/disable lpi mode
    if (eai_ret != EAI_SUCCESS) {
        EAI_LOG("eai_init fail, result = %d\n", eai_ret);
        return EAI_RUN_EAI_FAILURE;
    }

    eai_ret = eai_get_property(context->eai_handle, EAI_PROP_MODEL_META_INFO, &(context->eai_model_meta_info));
    if (eai_ret != EAI_SUCCESS) {
        EAI_LOG("eai_get_property(EAI_PROP_MODEL_META_INFO) fail, result = %d\n", eai_ret);
        return EAI_RUN_EAI_FAILURE;
    }

    char *model_name = NULL;
    eai_ret = eai_get_property(context->eai_handle, EAI_PROP_MODEL_NAME, &model_name);
    if (eai_ret == EAI_SUCCESS) {
        EAI_LOG("From eai_run : the model name is %s\n", model_name);
    }

    //model_type 0: fixed, 1: float
    bool is_enpu_supported = (context->eai_model_meta_info.model_type == 0 && context->eai_model_meta_info.enpu_ver >= 2);
    context->mla_usage = (context->use_enpu && is_enpu_supported) ? EAI_MLA_USAGE_TYPE_YES : EAI_MLA_USAGE_TYPE_NO;

    EAI_LOG("\neai: target enpu ver: %"PRIu32"\n", context->eai_model_meta_info.enpu_ver);
    EAI_LOG("\neai: enpu enable: %d\n", (int)context->mla_usage);

    // get scratch buffer info
    eai_ret = eai_get_property(context->eai_handle, EAI_PROP_SCRATCH_MEM, &scratch_memory);
    if (eai_ret != EAI_SUCCESS) {
        EAI_LOG("eai_get_property(EAI_PROP_SCRATCH_MEM) FAIL. result = %d\n", eai_ret);
        return EAI_RUN_EAI_FAILURE;
    }
    

    context->scratch_buffer_size = scratch_memory.memory_size;
    // context->scratch_buffer = (uint8_t *)mem_alloc(scratch_memory.memory_size, context->buffer_heap_type);
    context->scratch_buffer = (uint8_t *)malloc(scratch_memory.memory_size);

    if (!context->scratch_buffer){
        EAI_LOG("Could not allocate memory for scratch buffer\n");
        return EAI_RUN_MEMORY_ERROR;
    }
    
    scratch_memory.addr = context->scratch_buffer;
    scratch_memory.memory_type = EAI_MEM_TYPE_DDR;
    // set scratch buffer for eai api
    eai_ret = eai_set_property(context->eai_handle, EAI_PROP_SCRATCH_MEM, &scratch_memory);
    if (eai_ret != EAI_SUCCESS) {
        EAI_LOG("eai_set_property(EAI_PROP_SCRATCH_MEM) FAIL. result = %d\n", eai_ret);
        return EAI_RUN_EAI_FAILURE;
    }
    
    // get persistent buffer info
    eai_ret = eai_get_property(context->eai_handle, EAI_PROP_PERSISTENT_MEM, &persistent_memory);
    if (eai_ret != EAI_SUCCESS) {
        EAI_LOG("eai_get_property(EAI_PROP_PERSISTENT_MEM) FAIL. result = %d\n", eai_ret);
        return EAI_RUN_EAI_FAILURE;
    }

    context->persistent_buffer_size = persistent_memory.memory_size;
    context->persistent_buffer = (uint8_t *)malloc(persistent_memory.memory_size);
    // context->persistent_buffer = (uint8_t *)mem_alloc(persistent_memory.memory_size, context->hw_cfg_heap_type);
    
    if (!context->persistent_buffer){
        EAI_LOG("Could not allocate memory for persistent buffer\n");
        return EAI_RUN_MEMORY_ERROR;
    }

    persistent_memory.addr = context->persistent_buffer;
    persistent_memory.memory_type = EAI_MEM_TYPE_DDR;
    // set persistent buffer for eai api
    eai_ret = eai_set_property(context->eai_handle, EAI_PROP_PERSISTENT_MEM, &persistent_memory);
    if (eai_ret != EAI_SUCCESS) {
        EAI_LOG("eai_set_property(EAI_PROP_PERSISTENT_MEM) FAIL. result = %d\n", eai_ret);
        return EAI_RUN_EAI_FAILURE;
    }

    // Client perf config needs to be set before eai_apply
    eai_client_perf_config_t client_perf_config = EAI_CLIENT_PERF_CONFIG_INIT;
    client_perf_config.fps        = context->fps;
    client_perf_config.ftrt_ratio = context->ftrt_ratio;
    client_perf_config.priority = context->priority;
    client_perf_config.flags = 0x0;

    if ((eai_ret = eai_set_property(context->eai_handle, EAI_PROP_CLIENT_PERF_CFG, &client_perf_config)) != EAI_RUN_SUCCESS){
        EAI_LOG("Failed to set property EAI_PROP_CLIENT_PERF_CFG : result = %d \n", eai_ret);
        return EAI_RUN_EAI_FAILURE;
    }

    context->mla_usage = EAI_MLA_USAGE_TYPE_YES;
    // eai_ret = eai_set_property(context->eai_handle, EAI_PROP_MLA_USAGE, &context->mla_usage);
    // if (eai_ret != EAI_SUCCESS && eai_ret != EAI_MLA_NOT_AVAILABLE) {
    //     EAI_LOG("eai_set_property(EAI_PROP_MLA_USAGE) fail, result = %d\n", eai_ret);
    //     return -1;
    // }

    if (context->mla_usage == EAI_MLA_USAGE_TYPE_YES) {
        //set core affinity
        eai_mla_affinity_t client_affinity = {0};
        client_affinity.affinity = CORE_AFFINITY(context->affinity);
        client_affinity.core_selection = CORE_SELECTION(context->affinity);
        if ((eai_ret = eai_set_property(context->eai_handle, EAI_PROP_MLA_AFFINITY, &client_affinity)) != EAI_RUN_SUCCESS){
            EAI_LOG("Failed to set property EAI_PROP_MLA_AFFINITY : result = %d \n", eai_ret);
            return EAI_RUN_EAI_FAILURE;
        }
    }

    eai_ret = eai_apply(context->eai_handle);
    if (eai_ret != EAI_SUCCESS) {
        EAI_LOG("eai_apply FAIL. result = %d\n", eai_ret);
        return EAI_RUN_EAI_FAILURE;
    }

    for (int i = 0; i < 2; i ++){
        context->eai_buffers[i] = malloc_align(CONTEXT_BUFFER_ALIGNMENT, sizeof(eai_buffer_info_t) * context->tensor_count[i]);
    }

    return EAI_RUN_SUCCESS;
}

EAI_RUN_RESULT get_model_io(struct eai_run_context *context)
{
    EAI_RUN_RESULT ret = EAI_RUN_SUCCESS;
    EAI_RESULT eai_ret = EAI_SUCCESS;

    for (int i = 0; i < 2 && eai_ret == EAI_SUCCESS; i++)
    {
        eai_ports_info_t ports_info;
        ports_info.input_or_output = i;
        eai_ret = eai_get_property(context->eai_handle, EAI_PROP_PORTS_NUM, &ports_info);
        if (eai_ret != EAI_SUCCESS)
        {
            EAI_LOG("Failed eai_get_property(EAI_PROP_PORTS_NUM - inputs). result = %d\n", eai_ret);
            ret = EAI_RUN_EAI_FAILURE;
            break;
        }
        context->tensor_count[i] = ports_info.size;
        // context->tensors[i] = malloc_align(CONTEXT_BUFFER_ALIGNMENT, sizeof(eai_tensor_info_t) * ports_info.size);
        context->tensors[i] = malloc(sizeof(eai_tensor_info_t) * ports_info.size);
        if (context->tensors[i] == NULL){
            EAI_LOG("Failed to malloc tensor array\n");
            ret = EAI_RUN_MEMORY_ERROR;
            break;
        }

        for (unsigned int j = 0; j < ports_info.size; j++)
        {
            context->tensors[i][j].index = j;
            context->tensors[i][j].input_or_output = i;

            eai_ret = eai_get_property(context->eai_handle, EAI_PROP_TENSOR_INFO, &(context->tensors[i][j]));
            EAI_LOG("user scratch pointer context->tensors[%d][%d]: %p\n",i,j,context->tensors[i][j].address)
            if (eai_ret != EAI_SUCCESS)
            {
                EAI_LOG("Failed eai_get_property(EAI_PROP_TENSOR_SIZE_INFO - inputs). result = %d\n", eai_ret);
                ret = EAI_RUN_EAI_FAILURE;
                break;
            }
            if (!context->allocate_io_buf) {
                if (context->tensors[i][j].address == NULL) {
                    eai_ret = EAI_RESOURCE_FAILURE;
                    EAI_LOG("Input/Output not configured to use scratch, please use: -allocate_io\n");
                    ret = EAI_RUN_ERROR;
                    break;
                }
            }
        }
    }
    
    // if (eai_ret == EAI_SUCCESS && context->allocate_io_buf) {

    //     // get the hardware required alignment
    //     uint32_t buffer_alignment;
    //     eai_ret = eai_get_property(context->eai_handle, EAI_PROP_MODEL_ALIGNMENT, &buffer_alignment);

    //     if (eai_ret != EAI_SUCCESS){
    //         EAI_LOG("Failed to retrieve hardware buffer alignment\n");
    //         return EAI_RUN_ERROR;
    //     }

    //     size_t aligned_tensor_size = 0;
    //     for (int i = 0; i < 2 && eai_ret == EAI_SUCCESS; i++) {
    //         for (int j = 0; j < context->tensor_count[i] && eai_ret == EAI_SUCCESS; j++) {
    //             aligned_tensor_size = context->tensors[i][j].tensor_size + buffer_alignment; // Allocate extra memory for alignment since this buffer is being allocated from the user side
    //             context->tensors[i][j].address = malloc_align(buffer_alignment, aligned_tensor_size);

    //             if (context->tensors[i][j].address == NULL) {
    //                 EAI_LOG("Failed to allocate buffer for I/O\n");
    //                 eai_ret = EAI_RESOURCE_FAILURE;
    //                 return EAI_RUN_ERROR;
    //                 break;
    //             }

    //             //todo: register io buffer to the runtime if runtime is in the root pd
    //         }
    //     }
    // }
    return ret;
}

void print_model_io(struct eai_run_context *context)
{
    EAI_LOG("print model io \r\n");
    EAI_LOG("tensor count 0: %d \r\n", context->tensor_count[0]);
    EAI_LOG("tensor count 1: %d \r\n", context->tensor_count[1]);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < context->tensor_count[i]; j++) {
            if (i == 0) {
                EAI_LOG("\ninput: ");
            }
            else {
                EAI_LOG("output: ");
            }
            eai_tensor_info_t *tensor = &(context->tensors[i][j]);
            EAI_LOG("data type: %s\n", data_type_string[tensor->element_type]);
            EAI_LOG("dimension:");
            for (unsigned int k = 0; k < tensor->num_dims; k++) {
                EAI_LOG(" %"PRIu32"", (tensor->dims[k]));
            }
            EAI_LOG("\n\n");
        }
    }
}

int fill_io_batch(struct eai_run_context *context) {
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < context->tensor_count[i]; j++) {
            eai_tensor_info_t *tensor = &(context->tensors[i][j]);
            context->eai_buffers[i][j].index = j;
            context->eai_buffers[i][j].element_type = tensor->element_type;
            context->eai_buffers[i][j].addr = tensor->address;
            context->eai_buffers[i][j].buffer_size = tensor->tensor_size;
        }
    }
    context->eai_batch.num_inputs = context->tensor_count[0];
    context->eai_batch.num_outputs = context->tensor_count[1];
    context->eai_batch.inputs = &(context->eai_buffers[0][0]);
    context->eai_batch.outputs = &(context->eai_buffers[1][0]);
    return 0;
}

EAI_RUN_RESULT deinit(struct eai_run_context *context) {
    EAI_RUN_RESULT ret = EAI_RUN_SUCCESS;
    if (!context) {
        return EAI_RUN_EAI_FAILURE;
    }

    if (!context->eai_handle) {
        return EAI_RUN_ERROR;
    }

    EAI_RESULT eai_ret = eai_deinit(context->eai_handle);
    if (eai_ret != EAI_SUCCESS) {
        EAI_LOG("fail to deinit eai");
        return EAI_RUN_EAI_FAILURE;
    }
    if (context->scratch_buffer) {
        free(context->scratch_buffer);
    }    

    free_align(context->model_buffer);
    context->model_buffer = NULL;

    if (context->persistent_buffer) {
        free(context->persistent_buffer);
    }

    //free io buffers if allocated
    if (context->allocate_io_buf) {
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < context->tensor_count[i]; j++) {
                if(context->tensors[i][j].address) {
                    free_align(context->tensors[i][j].address);
                    context->tensors[i][j].address = NULL;
                }
            }
        }
    }

    return ret;
}

int generate_output_file_name(struct eai_run_context *context, char *full_path, int index) {
    char output_file_name[256];
    snprintf(output_file_name, 256, "output_%d.raw", index);
    if (context->output_path) {
        strcpy(full_path, context->output_path);
        if (context->output_path[strlen(context->output_path) - 1] != '/') {
            strcat(full_path, "/");
        }
    }
    strcat(full_path, output_file_name);
    return 0;
}

int initialize_o(struct eai_run_context *context)
{
    int ret = 0;
    for (int i = 0; i < context->tensor_count[1]; i++)
    {
        context->output_name[i] = (char *)malloc(MAX_FILE_PATH_LENGTH);
        if(context->output_name[i] == NULL) {
            ret = -1;
            break;
        }
        context->output_name[i][0] = 0;
        generate_output_file_name(context, context->output_name[i], i);
        context->io_file[1][i] = fopen(context->output_name[i], "wb");
        if (context->io_file[1][i] == NULL)
        {
            EAI_LOG("fail to open output file %s\n", context->output_name[i]);
            ret = -1;
            break;
        }
    }
    return ret;
}

int initialize_i(struct eai_run_context *context)
{
    int ret = 0;
    for (int i = 0; i < context->tensor_count[0]; i++)
    {
        context->io_file[0][i] = fopen(context->input_file[i], "rb");
        if (context->io_file[0][i] == NULL)
        {
            EAI_LOG("fail to open input file %s\n", context->input_file[i]);
            ret = -1;
            break;
        }
    }
    return ret;
}

int save_outputs(struct eai_run_context *context)
{
    int ret = 0;
    for (int i = 0; i < context->tensor_count[1]; i++)
    {
        eai_tensor_info_t *tensor = &context->tensors[1][i];
        if (!tensor || !tensor->address) {
            EAI_LOG("invalid i/o tensor!\n");
            ret = -1;
            break;
        }
        size_t write_size = fwrite(tensor->address, 1, tensor->tensor_size, context->io_file[1][i]);

        if (write_size != tensor->tensor_size)
        {
            ret = -1;
            break;
        }
    }
    return ret;
}
