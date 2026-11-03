"""Loss-free dense-GRPO execution rows; never extra original rollouts.

Called only after reward normalization and whole-rollout mask denominators have
been computed. Native get_batch pads the two-token sequence for TP/SP kernels.
"""
from copy import deepcopy


def append_alignment_padding(data: dict, anchor: int) -> tuple[int, int]:
    n = len(data['tokens'])
    if not 0 <= anchor < n or len(data['tokens'][anchor]) < 2:
        raise ValueError('Padding needs a real, nonempty trajectory anchor')
    vector_fields = {'rollout_log_probs', 'teacher_log_probs', 'ref_log_probs', 'log_probs'}
    scalar_fields = {'sample_indices', 'rollout_ids', 'rollout_mask_sums', 'source_names',
                     'round_number', 'metadata'}
    supported = vector_fields | scalar_fields | {
        'tokens', 'response_lengths', 'loss_masks', 'rewards', 'raw_reward',
        'truncated', 'total_lengths', 'alignment_padding', 'multimodal_train_inputs'}
    if set(data) - supported:
        raise ValueError('Alignment padding needs explicit field handling: '+str(sorted(set(data)-supported)))
    if any(len(values) != n for values in data.values()):
        raise ValueError('Training fields are not aligned before padding')
    if data.get('alignment_padding', [False]*n)[anchor]:
        raise ValueError('Cannot anchor padding to another padding row')
    # Construct every field before mutation so an unsupported representation
    # cannot leave a half-appended sample.
    row = {}
    for key, values in data.items():
        if key == 'tokens':
            row[key] = [int(values[anchor][0]), int(values[anchor][1])]
        elif key in vector_fields:
            row[key] = [0.0]
        elif key == 'loss_masks':
            row[key] = [0]
        elif key == 'response_lengths':
            row[key] = 1
        elif key == 'total_lengths':
            row[key] = 2
        elif key in ('rewards', 'raw_reward', 'truncated'):
            row[key] = 0
        elif key == 'multimodal_train_inputs':
            row[key] = None
        elif key == 'alignment_padding':
            row[key] = True
        else:
            # Share an EXISTING rollout ID and its ORIGINAL denominator. This
            # adds no new trajectory and has exactly zero numerator.
            row[key] = deepcopy(values[anchor])
    if 'alignment_padding' not in data:
        data['alignment_padding'] = [False]*n
        row['alignment_padding'] = True
    for key, value in row.items():
        data[key].append(value)
    return n, 2


def exclude_padding_metrics(data: dict) -> dict:
    """Return an unpadded metrics view; never modify the learner tensors.

    process_rollout_data already restores raw_reward to its original global
    shape, so it must not be filtered with this rank's local mask a second time.
    """
    mask = data.get('alignment_padding')
    if mask is None:
        return data
    kept = [i for i, padded in enumerate(mask) if not padded]
    view = {}
    global_keys = {'raw_reward', 'global_batch_sizes', 'num_microbatches', 'micro_batch_indices'}
    for key, value in data.items():
        if key == 'alignment_padding':
            continue
        if key not in global_keys and isinstance(value, (list, tuple)) and len(value) == len(mask):
            view[key] = [value[i] for i in kept]
        elif key == 'rollout_mask_sums':
            # This field is tensorized before reaching the logger.
            view[key] = value[kept]
        else:
            view[key] = value
    return view
