#include "vae_c_parameter.h"
#include "vae_c_tensor.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>


static int report_check(const char *name, int passed)
{
    if (passed) {
        printf("[PASS] %s\n", name);
        return 0;
    }

    fprintf(stderr, "[FAIL] %s\n", name);
    return 1;
}


static int shape_fields_equal(
    const VAECShape *left,
    const VAECShape *right
)
{
    size_t index;

    if (left == NULL || right == NULL ||
        left->rank != right->rank ||
        left->count != right->count) {

        return 0;
    }

    for (index = 0; index < VAE_C_TENSOR_MAX_RANK; index++) {
        if (left->dims[index] != right->dims[index] ||
            left->strides[index] != right->strides[index]) {

            return 0;
        }
    }

    return 1;
}


static int tensor_fields_equal(
    const VAECTensor *left,
    const VAECTensor *right
)
{
    return
        left != NULL &&
        right != NULL &&
        shape_fields_equal(&left->shape, &right->shape) &&
        left->data == right->data &&
        left->grad == right->grad &&
        left->owned_block == right->owned_block &&
        left->owned_bytes == right->owned_bytes &&
        left->owns_memory == right->owns_memory &&
        left->has_grad == right->has_grad;
}


static int parameter_fields_equal(
    const VAECParameter *left,
    const VAECParameter *right
)
{
    return
        left != NULL &&
        right != NULL &&
        shape_fields_equal(&left->shape, &right->shape) &&
        left->parameter_key == right->parameter_key &&
        left->values == right->values &&
        left->gradients == right->gradients &&
        left->adam_m == right->adam_m &&
        left->adam_v == right->adam_v &&
        left->owned_block == right->owned_block &&
        left->owned_bytes == right->owned_bytes;
}


static int all_zero(const float *values, size_t count)
{
    size_t index;

    for (index = 0; index < count; index++) {
        if (values[index] != 0.0f) {
            return 0;
        }
    }

    return 1;
}


static int all_equal(const float *values, size_t count, float expected)
{
    size_t index;

    for (index = 0; index < count; index++) {
        if (values[index] != expected) {
            return 0;
        }
    }

    return 1;
}


static int test_shape_contracts(void)
{
    const size_t dims_nchw[4] = {2, 4, 75, 75};
    const size_t dims_mini112[4] = {8, 16, 56, 56};
    const size_t dims_2d[2] = {8, 64};
    const size_t dims_1d[1] = {3136};
    const size_t dims_different[2] = {8, 63};
    const size_t indices_0[4] = {0, 0, 0, 0};
    const size_t indices_c1[4] = {0, 1, 0, 0};
    const size_t indices_n1[4] = {1, 0, 0, 0};
    const size_t indices_last[4] = {1, 3, 74, 74};
    VAECShape nchw;
    VAECShape nchw_copy;
    VAECShape mini112;
    VAECShape shape_2d;
    VAECShape shape_1d;
    VAECShape shape_1d_copy;
    VAECShape different;
    size_t offset;

    if (vae_c_shape_make(4, dims_nchw, &nchw) != VAE_C_STORAGE_OK ||
        vae_c_shape_make(4, dims_nchw, &nchw_copy) != VAE_C_STORAGE_OK ||
        vae_c_shape_make(4, dims_mini112, &mini112) !=
            VAE_C_STORAGE_OK ||
        vae_c_shape_make(2, dims_2d, &shape_2d) != VAE_C_STORAGE_OK ||
        vae_c_shape_make(1, dims_1d, &shape_1d) != VAE_C_STORAGE_OK ||
        vae_c_shape_make(1, dims_1d, &shape_1d_copy) !=
            VAE_C_STORAGE_OK ||
        vae_c_shape_make(2, dims_different, &different) !=
            VAE_C_STORAGE_OK) {

        return 0;
    }

    if (nchw.rank != 4 ||
        nchw.count != 45000 ||
        nchw.strides[0] != 22500 ||
        nchw.strides[1] != 5625 ||
        nchw.strides[2] != 75 ||
        nchw.strides[3] != 1 ||
        mini112.count != 401408 ||
        mini112.strides[0] != 50176 ||
        mini112.strides[1] != 3136 ||
        mini112.strides[2] != 56 ||
        mini112.strides[3] != 1 ||
        shape_2d.count != 512 ||
        shape_2d.strides[0] != 64 ||
        shape_2d.strides[1] != 1 ||
        shape_2d.dims[2] != 0 ||
        shape_2d.strides[2] != 0 ||
        shape_1d.count != 3136 ||
        shape_1d.strides[0] != 1 ||
        shape_1d.dims[1] != 0 ||
        shape_1d.strides[1] != 0 ||
        !vae_c_shape_equal(&shape_1d, &shape_1d_copy) ||
        !vae_c_shape_equal(&nchw, &nchw_copy) ||
        vae_c_shape_equal(&nchw, &shape_2d) ||
        vae_c_shape_equal(&shape_2d, &different)) {

        return 0;
    }

    if (vae_c_shape_offset(
            &nchw,
            indices_0,
            4,
            &offset) != VAE_C_STORAGE_OK ||
        offset != 0 ||
        vae_c_shape_offset(
            &nchw,
            indices_c1,
            4,
            &offset) != VAE_C_STORAGE_OK ||
        offset != 5625 ||
        vae_c_shape_offset(
            &nchw,
            indices_n1,
            4,
            &offset) != VAE_C_STORAGE_OK ||
        offset != 22500 ||
        vae_c_shape_offset(
            &nchw,
            indices_last,
            4,
            &offset) != VAE_C_STORAGE_OK ||
        offset != 44999) {

        return 0;
    }

    return 1;
}


static int test_shape_errors_and_strong_guarantee(void)
{
    const size_t valid_dims[2] = {8, 64};
    const size_t zero_dims[2] = {8, 0};
    const size_t overflow_dims[2] = {SIZE_MAX, 2};
    const size_t indices_ok[2] = {0, 0};
    const size_t indices_bad[2] = {8, 0};
    VAECShape shape;
    VAECShape snapshot;
    VAECShape malformed;
    VAECShape rank_zero;
    size_t offset;

    if (vae_c_shape_make(2, valid_dims, &shape) != VAE_C_STORAGE_OK) {
        return 0;
    }
    snapshot = shape;

    if (vae_c_shape_make(0, valid_dims, &shape) !=
            VAE_C_STORAGE_INVALID_RANK ||
        !shape_fields_equal(&shape, &snapshot) ||
        vae_c_shape_make(5, valid_dims, &shape) !=
            VAE_C_STORAGE_INVALID_RANK ||
        !shape_fields_equal(&shape, &snapshot) ||
        vae_c_shape_make(2, zero_dims, &shape) !=
            VAE_C_STORAGE_INVALID_DIMENSION ||
        !shape_fields_equal(&shape, &snapshot) ||
        vae_c_shape_make(2, overflow_dims, &shape) !=
            VAE_C_STORAGE_OVERFLOW ||
        !shape_fields_equal(&shape, &snapshot) ||
        vae_c_shape_make(2, NULL, &shape) !=
            VAE_C_STORAGE_INVALID_ARGUMENT ||
        !shape_fields_equal(&shape, &snapshot) ||
        vae_c_shape_make(2, valid_dims, NULL) !=
            VAE_C_STORAGE_INVALID_ARGUMENT) {

        return 0;
    }

    offset = 98765;
    if (vae_c_shape_offset(&shape, indices_ok, 1, &offset) !=
            VAE_C_STORAGE_SHAPE_MISMATCH ||
        offset != 98765 ||
        vae_c_shape_offset(&shape, indices_bad, 2, &offset) !=
            VAE_C_STORAGE_OUT_OF_RANGE ||
        offset != 98765 ||
        vae_c_shape_offset(&shape, NULL, 2, &offset) !=
            VAE_C_STORAGE_INVALID_ARGUMENT ||
        offset != 98765 ||
        vae_c_shape_offset(&shape, indices_ok, 2, NULL) !=
            VAE_C_STORAGE_INVALID_ARGUMENT) {

        return 0;
    }

    malformed = shape;
    malformed.strides[0]++;
    if (vae_c_shape_offset(
            &malformed,
            indices_ok,
            2,
            &offset) != VAE_C_STORAGE_INVALID_STATE ||
        offset != 98765) {

        return 0;
    }

    if (vae_c_shape_equal(&malformed, &shape)) {
        return 0;
    }

    malformed = shape;
    malformed.count++;
    if (vae_c_shape_equal(&malformed, &shape)) {
        return 0;
    }

    malformed = shape;
    malformed.rank = VAE_C_TENSOR_MAX_RANK + 1;
    if (vae_c_shape_equal(&malformed, &shape)) {
        return 0;
    }

    malformed = shape;
    malformed.dims[2] = 1;
    if (vae_c_shape_equal(&malformed, &shape)) {
        return 0;
    }

    malformed = shape;
    malformed.strides[2] = 1;
    if (vae_c_shape_equal(&malformed, &shape)) {
        return 0;
    }

    vae_c_shape_init(&rank_zero);

    return !vae_c_shape_equal(NULL, &shape) &&
           !vae_c_shape_equal(&shape, NULL) &&
           !vae_c_shape_equal(&rank_zero, &rank_zero);
}


static int test_tensor_allocation_and_zeroing(void)
{
    const size_t dims[2] = {2, 3};
    VAECShape shape;
    VAECTensor data_only;
    VAECTensor with_grad;
    VAECTensor empty;
    VAECTensor malformed_empty;
    size_t index;

    vae_c_tensor_init(&data_only);
    vae_c_tensor_init(&with_grad);
    vae_c_tensor_init(&empty);
    vae_c_tensor_init(&malformed_empty);

    if (!vae_c_tensor_is_empty(&data_only) ||
        !shape_fields_equal(&data_only.shape, &empty.shape) ||
        vae_c_shape_make(2, dims, &shape) != VAE_C_STORAGE_OK ||
        vae_c_tensor_allocate(
            &data_only,
            &shape,
            false) != VAE_C_STORAGE_OK ||
        vae_c_tensor_is_empty(&data_only) ||
        data_only.data == NULL ||
        data_only.data != data_only.owned_block ||
        data_only.grad != NULL ||
        !data_only.owns_memory ||
        data_only.has_grad ||
        data_only.owned_bytes != shape.count * sizeof(float) ||
        !all_zero(data_only.data, shape.count)) {

        vae_c_tensor_free(&data_only);
        return 0;
    }

    data_only.data[0] = 7.0f;
    if (vae_c_tensor_zero_grad(&data_only) !=
            VAE_C_STORAGE_GRADIENT_NOT_AVAILABLE ||
        data_only.data[0] != 7.0f ||
        vae_c_tensor_zero_data(&data_only) != VAE_C_STORAGE_OK ||
        !all_zero(data_only.data, shape.count)) {

        vae_c_tensor_free(&data_only);
        return 0;
    }

    if (vae_c_tensor_allocate(
            &with_grad,
            &shape,
            true) != VAE_C_STORAGE_OK ||
        with_grad.data != with_grad.owned_block ||
        with_grad.grad != with_grad.data + shape.count ||
        !with_grad.owns_memory ||
        !with_grad.has_grad ||
        with_grad.owned_bytes !=
            2 * shape.count * sizeof(float) ||
        !all_zero(with_grad.data, shape.count) ||
        !all_zero(with_grad.grad, shape.count)) {

        vae_c_tensor_free(&data_only);
        vae_c_tensor_free(&with_grad);
        return 0;
    }

    with_grad.data[0] = 3.0f;
    with_grad.grad[0] = 5.0f;
    if (with_grad.data[0] != 3.0f ||
        with_grad.grad[0] != 5.0f) {

        vae_c_tensor_free(&data_only);
        vae_c_tensor_free(&with_grad);
        return 0;
    }

    for (index = 0; index < shape.count; index++) {
        with_grad.data[index] = 2.0f;
        with_grad.grad[index] = 4.0f;
    }

    if (vae_c_tensor_zero_data(&with_grad) != VAE_C_STORAGE_OK ||
        !all_zero(with_grad.data, shape.count) ||
        !all_equal(with_grad.grad, shape.count, 4.0f)) {

        vae_c_tensor_free(&data_only);
        vae_c_tensor_free(&with_grad);
        return 0;
    }

    for (index = 0; index < shape.count; index++) {
        with_grad.data[index] = 6.0f;
    }

    if (vae_c_tensor_zero_grad(&with_grad) != VAE_C_STORAGE_OK ||
        !all_zero(with_grad.grad, shape.count) ||
        !all_equal(with_grad.data, shape.count, 6.0f)) {

        vae_c_tensor_free(&data_only);
        vae_c_tensor_free(&with_grad);
        return 0;
    }

    vae_c_tensor_free(&data_only);
    vae_c_tensor_free(&data_only);
    vae_c_tensor_free(&with_grad);
    vae_c_tensor_free(&with_grad);

    if (!vae_c_tensor_is_empty(&data_only) ||
        !vae_c_tensor_is_empty(&with_grad) ||
        vae_c_tensor_zero_data(&data_only) != VAE_C_STORAGE_OK ||
        vae_c_tensor_zero_grad(&with_grad) != VAE_C_STORAGE_OK) {

        return 0;
    }

    malformed_empty.data = (float *)(uintptr_t)1;
    if (vae_c_tensor_is_empty(&malformed_empty)) {
        return 0;
    }

    vae_c_tensor_init(&malformed_empty);
    malformed_empty.owned_bytes = 1;
    if (vae_c_tensor_is_empty(&malformed_empty)) {
        return 0;
    }

    vae_c_tensor_init(&malformed_empty);
    malformed_empty.has_grad = true;
    return !vae_c_tensor_is_empty(&malformed_empty);
}


static int test_tensor_errors_and_overflows(void)
{
    const size_t small_dims[1] = {8};
    const size_t product_overflow_dims[2] = {SIZE_MAX, 2};
    const size_t bytes_overflow_dims[1] =
        {SIZE_MAX / sizeof(float) + 1};
    const size_t grad_elements_overflow_dims[1] =
        {SIZE_MAX / 2 + 1};
    const size_t grad_bytes_overflow_dims[1] =
        {SIZE_MAX / (2 * sizeof(float)) + 1};
    VAECShape small_shape;
    VAECShape invalid_shape;
    VAECTensor tensor;
    VAECTensor snapshot;

    if (vae_c_shape_make(1, small_dims, &small_shape) !=
        VAE_C_STORAGE_OK) {

        return 0;
    }

    vae_c_tensor_init(&tensor);
    if (vae_c_tensor_allocate(
            &tensor,
            &small_shape,
            true) != VAE_C_STORAGE_OK) {

        return 0;
    }
    snapshot = tensor;

    if (vae_c_tensor_allocate(
            &tensor,
            &small_shape,
            false) != VAE_C_STORAGE_OBJECT_NOT_EMPTY ||
        !tensor_fields_equal(&tensor, &snapshot)) {

        vae_c_tensor_free(&tensor);
        return 0;
    }

    vae_c_tensor_free(&tensor);
    vae_c_shape_init(&invalid_shape);

    if (vae_c_tensor_allocate(
            &tensor,
            &invalid_shape,
            false) != VAE_C_STORAGE_INVALID_RANK ||
        !vae_c_tensor_is_empty(&tensor)) {

        return 0;
    }

    vae_c_shape_init(&invalid_shape);
    invalid_shape.rank = 5;
    if (vae_c_tensor_allocate(
            &tensor,
            &invalid_shape,
            false) != VAE_C_STORAGE_INVALID_RANK ||
        !vae_c_tensor_is_empty(&tensor)) {

        return 0;
    }

    vae_c_shape_init(&invalid_shape);
    invalid_shape.rank = 1;
    invalid_shape.dims[0] = 0;
    if (vae_c_tensor_allocate(
            &tensor,
            &invalid_shape,
            false) != VAE_C_STORAGE_INVALID_DIMENSION ||
        !vae_c_tensor_is_empty(&tensor)) {

        return 0;
    }

    vae_c_shape_init(&invalid_shape);
    invalid_shape.rank = 2;
    invalid_shape.dims[0] = product_overflow_dims[0];
    invalid_shape.dims[1] = product_overflow_dims[1];
    if (vae_c_tensor_allocate(
            &tensor,
            &invalid_shape,
            false) != VAE_C_STORAGE_OVERFLOW ||
        !vae_c_tensor_is_empty(&tensor)) {

        return 0;
    }

    if (vae_c_shape_make(
            1,
            bytes_overflow_dims,
            &invalid_shape) != VAE_C_STORAGE_OK ||
        vae_c_tensor_allocate(
            &tensor,
            &invalid_shape,
            false) != VAE_C_STORAGE_OVERFLOW ||
        !vae_c_tensor_is_empty(&tensor)) {

        return 0;
    }

    if (vae_c_shape_make(
            1,
            grad_elements_overflow_dims,
            &invalid_shape) != VAE_C_STORAGE_OK ||
        vae_c_tensor_allocate(
            &tensor,
            &invalid_shape,
            true) != VAE_C_STORAGE_OVERFLOW ||
        !vae_c_tensor_is_empty(&tensor)) {

        return 0;
    }

    if (vae_c_shape_make(
            1,
            grad_bytes_overflow_dims,
            &invalid_shape) != VAE_C_STORAGE_OK ||
        vae_c_tensor_allocate(
            &tensor,
            &invalid_shape,
            true) != VAE_C_STORAGE_OVERFLOW ||
        !vae_c_tensor_is_empty(&tensor)) {

        return 0;
    }

    return
        vae_c_tensor_allocate(NULL, &small_shape, false) ==
            VAE_C_STORAGE_INVALID_ARGUMENT &&
        vae_c_tensor_allocate(&tensor, NULL, false) ==
            VAE_C_STORAGE_INVALID_ARGUMENT &&
        vae_c_tensor_zero_data(NULL) ==
            VAE_C_STORAGE_INVALID_ARGUMENT &&
        vae_c_tensor_zero_grad(NULL) ==
            VAE_C_STORAGE_INVALID_ARGUMENT &&
        !vae_c_tensor_is_empty(NULL);
}


static int test_contiguous_views(void)
{
    const size_t source_dims[4] = {2, 4, 75, 75};
    const size_t reshape_dims[2] = {2, 22500};
    const size_t batch_dims[4] = {8, 4, 75, 75};
    const size_t prefix_dims[4] = {3, 4, 75, 75};
    const size_t region_dims[1] = {100};
    VAECShape source_shape;
    VAECShape reshape_shape;
    VAECShape batch_shape;
    VAECShape prefix_shape;
    VAECShape region_shape;
    VAECTensor source;
    VAECTensor reshape_view;
    VAECTensor batch_source;
    VAECTensor prefix_view;
    VAECTensor region_view;

    vae_c_tensor_init(&source);
    vae_c_tensor_init(&reshape_view);
    vae_c_tensor_init(&batch_source);
    vae_c_tensor_init(&prefix_view);
    vae_c_tensor_init(&region_view);

    if (vae_c_shape_make(4, source_dims, &source_shape) !=
            VAE_C_STORAGE_OK ||
        vae_c_shape_make(2, reshape_dims, &reshape_shape) !=
            VAE_C_STORAGE_OK ||
        vae_c_tensor_allocate(&source, &source_shape, true) !=
            VAE_C_STORAGE_OK ||
        vae_c_tensor_make_contiguous_view(
            &source,
            0,
            &reshape_shape,
            &reshape_view) != VAE_C_STORAGE_OK ||
        reshape_view.data != source.data ||
        reshape_view.grad != source.grad ||
        reshape_view.owns_memory ||
        reshape_view.owned_block != NULL ||
        reshape_view.owned_bytes != 0 ||
        !reshape_view.has_grad) {

        vae_c_tensor_free(&reshape_view);
        vae_c_tensor_free(&source);
        return 0;
    }

    reshape_view.data[10] = 11.0f;
    reshape_view.grad[20] = 12.0f;
    if (source.data[10] != 11.0f || source.grad[20] != 12.0f) {
        vae_c_tensor_free(&reshape_view);
        vae_c_tensor_free(&source);
        return 0;
    }

    vae_c_tensor_free(&reshape_view);
    if (source.data[10] != 11.0f || !source.owns_memory) {
        vae_c_tensor_free(&source);
        return 0;
    }

    if (vae_c_shape_make(4, batch_dims, &batch_shape) !=
            VAE_C_STORAGE_OK ||
        vae_c_shape_make(4, prefix_dims, &prefix_shape) !=
            VAE_C_STORAGE_OK ||
        vae_c_shape_make(1, region_dims, &region_shape) !=
            VAE_C_STORAGE_OK ||
        vae_c_tensor_allocate(&batch_source, &batch_shape, true) !=
            VAE_C_STORAGE_OK ||
        vae_c_tensor_make_contiguous_view(
            &batch_source,
            0,
            &prefix_shape,
            &prefix_view) != VAE_C_STORAGE_OK ||
        prefix_view.shape.count != 67500 ||
        prefix_view.data != batch_source.data ||
        prefix_view.grad != batch_source.grad ||
        vae_c_tensor_make_contiguous_view(
            &batch_source,
            123,
            &region_shape,
            &region_view) != VAE_C_STORAGE_OK ||
        region_view.data != batch_source.data + 123 ||
        region_view.grad != batch_source.grad + 123) {

        vae_c_tensor_free(&region_view);
        vae_c_tensor_free(&prefix_view);
        vae_c_tensor_free(&batch_source);
        vae_c_tensor_free(&source);
        return 0;
    }

    vae_c_tensor_free(&region_view);
    vae_c_tensor_free(&prefix_view);
    vae_c_tensor_free(&batch_source);
    vae_c_tensor_free(&source);
    return 1;
}


static int test_view_errors_and_strong_guarantee(void)
{
    const size_t source_dims[1] = {10};
    const size_t view_dims[1] = {1};
    const size_t large_dims[1] = {11};
    VAECShape source_shape;
    VAECShape view_shape;
    VAECShape large_shape;
    VAECShape invalid_shape;
    VAECTensor source;
    VAECTensor view;
    VAECTensor nonempty_view;
    VAECTensor snapshot;
    VAECTensor empty_source;

    vae_c_tensor_init(&source);
    vae_c_tensor_init(&view);
    vae_c_tensor_init(&nonempty_view);
    vae_c_tensor_init(&empty_source);

    if (vae_c_shape_make(1, source_dims, &source_shape) !=
            VAE_C_STORAGE_OK ||
        vae_c_shape_make(1, view_dims, &view_shape) !=
            VAE_C_STORAGE_OK ||
        vae_c_shape_make(1, large_dims, &large_shape) !=
            VAE_C_STORAGE_OK ||
        vae_c_tensor_allocate(&source, &source_shape, true) !=
            VAE_C_STORAGE_OK ||
        vae_c_tensor_allocate(&nonempty_view, &view_shape, false) !=
            VAE_C_STORAGE_OK) {

        vae_c_tensor_free(&source);
        vae_c_tensor_free(&nonempty_view);
        return 0;
    }

    snapshot = source;
    if (vae_c_tensor_make_contiguous_view(
            &source,
            0,
            &view_shape,
            &source) != VAE_C_STORAGE_INVALID_ARGUMENT ||
        !tensor_fields_equal(&source, &snapshot)) {

        vae_c_tensor_free(&source);
        vae_c_tensor_free(&nonempty_view);
        return 0;
    }

    snapshot = nonempty_view;
    if (vae_c_tensor_make_contiguous_view(
            &source,
            0,
            &view_shape,
            &nonempty_view) != VAE_C_STORAGE_OBJECT_NOT_EMPTY ||
        !tensor_fields_equal(&nonempty_view, &snapshot)) {

        vae_c_tensor_free(&source);
        vae_c_tensor_free(&nonempty_view);
        return 0;
    }

    if (vae_c_tensor_make_contiguous_view(
            &empty_source,
            0,
            &view_shape,
            &view) != VAE_C_STORAGE_INVALID_STATE ||
        !vae_c_tensor_is_empty(&view) ||
        vae_c_tensor_make_contiguous_view(
            &source,
            0,
            &large_shape,
            &view) != VAE_C_STORAGE_OUT_OF_RANGE ||
        !vae_c_tensor_is_empty(&view) ||
        vae_c_tensor_make_contiguous_view(
            &source,
            source.shape.count,
            &view_shape,
            &view) != VAE_C_STORAGE_OUT_OF_RANGE ||
        !vae_c_tensor_is_empty(&view) ||
        vae_c_tensor_make_contiguous_view(
            &source,
            SIZE_MAX,
            &view_shape,
            &view) != VAE_C_STORAGE_OVERFLOW ||
        !vae_c_tensor_is_empty(&view)) {

        vae_c_tensor_free(&source);
        vae_c_tensor_free(&nonempty_view);
        return 0;
    }

    vae_c_shape_init(&invalid_shape);
    if (vae_c_tensor_make_contiguous_view(
            &source,
            0,
            &invalid_shape,
            &view) != VAE_C_STORAGE_INVALID_RANK ||
        !vae_c_tensor_is_empty(&view) ||
        vae_c_tensor_make_contiguous_view(
            NULL,
            0,
            &view_shape,
            &view) != VAE_C_STORAGE_INVALID_ARGUMENT ||
        vae_c_tensor_make_contiguous_view(
            &source,
            0,
            NULL,
            &view) != VAE_C_STORAGE_INVALID_ARGUMENT ||
        vae_c_tensor_make_contiguous_view(
            &source,
            0,
            &view_shape,
            NULL) != VAE_C_STORAGE_INVALID_ARGUMENT) {

        vae_c_tensor_free(&source);
        vae_c_tensor_free(&nonempty_view);
        return 0;
    }

    vae_c_tensor_free(&nonempty_view);
    vae_c_tensor_free(&source);
    return 1;
}


static int test_parameter_allocation_and_zeroing(void)
{
    const size_t conv_dims[4] = {32, 4, 3, 3};
    VAECShape shape;
    VAECParameter parameter;
    VAECParameter malformed_empty;
    size_t index;

    vae_c_parameter_init(&parameter);
    vae_c_parameter_init(&malformed_empty);

    if (!vae_c_parameter_is_empty(&parameter) ||
        vae_c_shape_make(4, conv_dims, &shape) != VAE_C_STORAGE_OK ||
        shape.count != 1152 ||
        vae_c_parameter_allocate(
            &parameter,
            &shape,
            UINT64_C(77)) != VAE_C_STORAGE_OK ||
        vae_c_parameter_is_empty(&parameter) ||
        parameter.parameter_key != UINT64_C(77) ||
        parameter.values != parameter.owned_block ||
        parameter.gradients != parameter.values + shape.count ||
        parameter.adam_m != parameter.gradients + shape.count ||
        parameter.adam_v != parameter.adam_m + shape.count ||
        parameter.owned_bytes !=
            4 * shape.count * sizeof(float) ||
        !all_zero(parameter.values, shape.count) ||
        !all_zero(parameter.gradients, shape.count) ||
        !all_zero(parameter.adam_m, shape.count) ||
        !all_zero(parameter.adam_v, shape.count)) {

        vae_c_parameter_free(&parameter);
        return 0;
    }

    parameter.values[0] = 1.0f;
    parameter.gradients[1] = 2.0f;
    parameter.adam_m[2] = 3.0f;
    parameter.adam_v[3] = 4.0f;

    if (parameter.values[0] != 1.0f ||
        parameter.gradients[1] != 2.0f ||
        parameter.adam_m[2] != 3.0f ||
        parameter.adam_v[3] != 4.0f) {

        vae_c_parameter_free(&parameter);
        return 0;
    }

    for (index = 0; index < shape.count; index++) {
        parameter.values[index] = 5.0f;
        parameter.gradients[index] = 6.0f;
        parameter.adam_m[index] = 7.0f;
        parameter.adam_v[index] = 8.0f;
    }

    if (vae_c_parameter_zero_grad(&parameter) != VAE_C_STORAGE_OK ||
        !all_equal(parameter.values, shape.count, 5.0f) ||
        !all_zero(parameter.gradients, shape.count) ||
        !all_equal(parameter.adam_m, shape.count, 7.0f) ||
        !all_equal(parameter.adam_v, shape.count, 8.0f)) {

        vae_c_parameter_free(&parameter);
        return 0;
    }

    for (index = 0; index < shape.count; index++) {
        parameter.gradients[index] = 9.0f;
    }

    if (vae_c_parameter_zero_optimizer_state(&parameter) !=
            VAE_C_STORAGE_OK ||
        !all_equal(parameter.values, shape.count, 5.0f) ||
        !all_equal(parameter.gradients, shape.count, 9.0f) ||
        !all_zero(parameter.adam_m, shape.count) ||
        !all_zero(parameter.adam_v, shape.count)) {

        vae_c_parameter_free(&parameter);
        return 0;
    }

    vae_c_parameter_free(&parameter);
    vae_c_parameter_free(&parameter);

    if (!vae_c_parameter_is_empty(&parameter) ||
        parameter.parameter_key != UINT64_C(0) ||
        vae_c_parameter_zero_grad(&parameter) != VAE_C_STORAGE_OK ||
        vae_c_parameter_zero_optimizer_state(&parameter) !=
            VAE_C_STORAGE_OK ||
        vae_c_parameter_allocate(
            &parameter,
            &shape,
            UINT64_C(0)) != VAE_C_STORAGE_OK ||
        parameter.parameter_key != UINT64_C(0)) {

        vae_c_parameter_free(&parameter);
        return 0;
    }

    vae_c_parameter_free(&parameter);

    malformed_empty.parameter_key = UINT64_C(1);
    if (vae_c_parameter_is_empty(&malformed_empty)) {
        return 0;
    }

    vae_c_parameter_init(&malformed_empty);
    malformed_empty.values = (float *)(uintptr_t)1;
    if (vae_c_parameter_is_empty(&malformed_empty)) {
        return 0;
    }

    vae_c_parameter_init(&malformed_empty);
    malformed_empty.owned_bytes = 1;
    return !vae_c_parameter_is_empty(&malformed_empty);
}


static int test_parameter_errors_and_strong_guarantee(void)
{
    const size_t valid_dims[1] = {8};
    const size_t count_overflow_dims[1] = {SIZE_MAX / 4 + 1};
    const size_t byte_overflow_dims[1] =
        {SIZE_MAX / (4 * sizeof(float)) + 1};
    VAECShape valid_shape;
    VAECShape invalid_shape;
    VAECParameter parameter;
    VAECParameter snapshot;

    vae_c_parameter_init(&parameter);

    if (vae_c_shape_make(1, valid_dims, &valid_shape) !=
            VAE_C_STORAGE_OK ||
        vae_c_parameter_allocate(
            &parameter,
            &valid_shape,
            UINT64_C(9)) != VAE_C_STORAGE_OK) {

        return 0;
    }

    snapshot = parameter;
    if (vae_c_parameter_allocate(
            &parameter,
            &valid_shape,
            UINT64_C(10)) != VAE_C_STORAGE_OBJECT_NOT_EMPTY ||
        !parameter_fields_equal(&parameter, &snapshot)) {

        vae_c_parameter_free(&parameter);
        return 0;
    }

    vae_c_parameter_free(&parameter);
    vae_c_shape_init(&invalid_shape);

    if (vae_c_parameter_allocate(
            &parameter,
            &invalid_shape,
            UINT64_C(1)) != VAE_C_STORAGE_INVALID_RANK ||
        !vae_c_parameter_is_empty(&parameter)) {

        return 0;
    }

    if (vae_c_shape_make(
            1,
            count_overflow_dims,
            &invalid_shape) != VAE_C_STORAGE_OK ||
        vae_c_parameter_allocate(
            &parameter,
            &invalid_shape,
            UINT64_C(1)) != VAE_C_STORAGE_OVERFLOW ||
        !vae_c_parameter_is_empty(&parameter)) {

        return 0;
    }

    if (vae_c_shape_make(
            1,
            byte_overflow_dims,
            &invalid_shape) != VAE_C_STORAGE_OK ||
        vae_c_parameter_allocate(
            &parameter,
            &invalid_shape,
            UINT64_C(1)) != VAE_C_STORAGE_OVERFLOW ||
        !vae_c_parameter_is_empty(&parameter)) {

        return 0;
    }

    return
        vae_c_parameter_allocate(NULL, &valid_shape, UINT64_C(0)) ==
            VAE_C_STORAGE_INVALID_ARGUMENT &&
        vae_c_parameter_allocate(&parameter, NULL, UINT64_C(0)) ==
            VAE_C_STORAGE_INVALID_ARGUMENT &&
        vae_c_parameter_zero_grad(NULL) ==
            VAE_C_STORAGE_INVALID_ARGUMENT &&
        vae_c_parameter_zero_optimizer_state(NULL) ==
            VAE_C_STORAGE_INVALID_ARGUMENT &&
        !vae_c_parameter_is_empty(NULL);
}


static int test_status_names_and_null_lifecycle(void)
{
    size_t status;

    for (status = VAE_C_STORAGE_OK;
         status <= VAE_C_STORAGE_GRADIENT_NOT_AVAILABLE;
         status++) {

        if (strcmp(
                vae_c_storage_status_name((VAECStorageStatus)status),
                "unknown") == 0) {

            return 0;
        }
    }

    vae_c_shape_init(NULL);
    vae_c_tensor_init(NULL);
    vae_c_tensor_free(NULL);
    vae_c_parameter_init(NULL);
    vae_c_parameter_free(NULL);

    return
        strcmp(
            vae_c_storage_status_name((VAECStorageStatus)999),
            "unknown") == 0;
}


int main(void)
{
    int failures = 0;

    failures += report_check(
        "shape_contracts",
        test_shape_contracts()
    );
    failures += report_check(
        "shape_errors_and_strong_guarantee",
        test_shape_errors_and_strong_guarantee()
    );
    failures += report_check(
        "tensor_allocation_and_zeroing",
        test_tensor_allocation_and_zeroing()
    );
    failures += report_check(
        "tensor_errors_and_overflows",
        test_tensor_errors_and_overflows()
    );
    failures += report_check(
        "contiguous_views",
        test_contiguous_views()
    );
    failures += report_check(
        "view_errors_and_strong_guarantee",
        test_view_errors_and_strong_guarantee()
    );
    failures += report_check(
        "parameter_allocation_and_zeroing",
        test_parameter_allocation_and_zeroing()
    );
    failures += report_check(
        "parameter_errors_and_strong_guarantee",
        test_parameter_errors_and_strong_guarantee()
    );
    failures += report_check(
        "status_names_and_null_lifecycle",
        test_status_names_and_null_lifecycle()
    );

    if (failures != 0) {
        fprintf(
            stderr,
            "%d VAE C TENSOR/PARAMETER TEST(S) FAILED\n",
            failures
        );
        return 1;
    }

    printf("ALL VAE C TENSOR/PARAMETER TESTS PASSED\n");
    return 0;
}
