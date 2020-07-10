import sys
import os


def load_jiant_model(
    ckpt_path,
    jiant_dir="../jiant-russian/",
    pretrain_tasks="all",
    target_tasks="all",
    config_file="../jiant-russian/jiant/config/superglue_bert.conf",
    jiant_prefix=None,
    jiant_data=None,
    word_embeds_file="None",
    cuda_device=0
):
    """
    Load jiant pretrained model and tasks.

    Parameters
    ----------
    ckpt_path : str
        Path to pretrained model's checkpoint.
    
    jiant_dir : str, default="../jiant-russian/"
        Directory with model_dir, data and jiant.
    
    pretrain_tasks : str
        Names of tasks for pretrain. One of
        {"danetqa", "rcb", "parus", "muserc", "rucos", "terra", "russe", "rwsd", "all"}
    
    target_tasks : str
        Names of target tasks. One of
        {"danetqa", "rcb", "parus", "muserc", "rucos", "terra", "russe", "rwsd", "all"}
    
    config_file : str, default="../jiant-russian/jiant/config/superglue_bert.conf"
        Path to configuration file of model.
    
    jiant_prefix : str or None, default=None
        JIANT_PROJECT_PREFIX from user_config.sh file.
        If None jiant_prefix = jiant_dir/model_dir/.
    
    jiant_data : str or None, default=None
        JIANT_DATA_DIR from user_config.sh file.
        If None jiant_data = jiant_dir/combined/.

    word_embeds_file : str, default="None"
        WORD_EMBS_FILE from user_config.sh file.
        Path to word embeddings. May be not used.

    cuda_device : int or None, default=0
        Cuda device number or None.
    
    Returns
    ----------
    tuple with following format
        model - jiant model

        embedder_model - extracted embedder model
            (usually model from https://github.com/huggingface/transformers)

        tokenizer - extracted text tokenizer
            (usually model from https://github.com/huggingface/transformers)

        pretrain_tasks - list of tasks objects for pretrain
            (for example [jiant.tasks.tasks.TERRaSuperGLUETask])
            Each task should be inherited from jiant.tasks.tasks.Task

        target_tasks - list of tasks objects for predict
            (for example [jiant.tasks.tasks.TERRaSuperGLUETask])
            Each task should be inherited from jiant.tasks.tasks.Task
    """
    sys.path.append(jiant_dir)

    from jiant.__main__ import (
        handle_arguments,
        config,
        check_arg_name,
        initial_setup,
        build_tasks,
        build_model,
        load_model_state
    )

    if jiant_prefix is None:
        jiant_prefix = f"{jiant_dir}/model_dir/"
    if jiant_data is None:
        jiant_data = f"{jiant_dir}/combined/"
    
    # Setup jiant environ (as in user_config.sh)
    os.environ["JIANT_PROJECT_PREFIX"] = jiant_prefix
    os.environ["JIANT_DATA_DIR"] = jiant_data
    os.environ["WORD_EMBS_FILE"] = "None"
    cl_arguments = []
    cl_args = handle_arguments(cl_arguments)
    cl_args.config_file = config_file
    args = config.params_from_file(cl_args.config_file, cl_args.overrides)
    # Check for deprecated arg names
    check_arg_name(args)
    args, seed = initial_setup(args, cl_args)
    args.pretrain_tasks = pretrain_tasks
    args.target_tasks = target_tasks
    pretrain_tasks, target_tasks, vocab, word_embs = build_tasks(args, cuda_device)
    tasks = sorted(set(pretrain_tasks + target_tasks), key=lambda x: x.name)
    model = build_model(args, vocab, word_embs, tasks, cuda_device)
    load_model_state(model, ckpt_path, cuda_device, skip_task_models=[], strict=False)

    tokenizer = model.sent_encoder._text_field_embedder.tokenizer
    embedder_model = model.sent_encoder._text_field_embedder.model
    
    return model, embedder_model, tokenizer, pretrain_tasks, target_tasks


def _device_iterator(evaluate, generator, cuda_device):
    """Move batch to device (if needed)"""
    for batch_idx, batch in enumerate(generator):
        if isinstance(cuda_device, int):
            batch = evaluate.move_to_device(batch, cuda_device)
        yield (batch_idx, batch)


def get_dataloader(
    task,
    split,
    jiant_dir="../jiant-russian/",
    batch_size=16,
    shuffle=False,
    cuda_device=0
):
    """
    Extract dataloader from task
    
    Parameters
    ----------
    jiant_dir : str, default="../jiant-russian/"
        Directory with model_dir, data and jiant.
    
    task : jiant.tasks.tasks.Task
        Task object.
    
    split : str
        Part of data : train, val, test.

    batch_size : int, default=16
        Batch size.

    shuffle : bool, default=False
        If need shuffle dataset.
    
    cuda_device : int or None, default=0
        Cuda device number or None.
    
    Returns
    ----------
    batch : generator
        Batch generator.
    """
    sys.path.append(jiant_dir)

    from jiant.__main__ import evaluate

    dataset = getattr(task, "%s_data" % split)
    iterator = evaluate.BasicIterator(batch_size)
    generator = iterator(dataset, num_epochs=1, shuffle=shuffle)
    return _device_iterator(evaluate, generator, cuda_device)
