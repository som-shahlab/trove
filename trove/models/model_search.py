import numpy as np
from itertools import product
from ..analysis.metrics import score_sequences, tokens_to_sequences
from ..analysis.error_analysis import mv
from sklearn.metrics import (
    precision_score, recall_score,
    f1_score, accuracy_score,
    precision_recall_fscore_support
)


def sample_param_grid(param_grid, seed):
    """ Sample parameter grid

    :param param_grid:
    :param seed:
    :return:
    """
    rstate = np.random.get_state()
    np.random.seed(seed)
    params = list(product(*[param_grid[name] for name in param_grid]))
    np.random.shuffle(params)
    np.random.set_state(rstate)
    return params


def compute_metrics(y_gold, y_pred, average='binary'):
    """

    :param y_gold:
    :param y_pred:
    :param average:
    :return:
    """
    return {
        'accuracy': accuracy_score(y_gold, y_pred),
        'precision': precision_score(y_gold, y_pred, average=average),
        'recall': recall_score(y_gold, y_pred, average=average),
        'f1': f1_score(y_gold, y_pred, average=average)
    }


def grid_search_span(model_class,
                     model_class_init,
                     param_grid,
                     train=None,
                     dev=None,
                     n_model_search=5,
                     val_metric='f1',
                     seed=1234,
                     verbose=True):
    """Simple grid search helper function

    """
    L_train, Y_train = train if len(train) == 2 else (train[0], None)
    L_dev, Y_dev = dev

    # sample configs
    params = sample_param_grid(param_grid, seed)[:n_model_search]

    defaults = {'optimizer': 'adam', 'seed': seed}
    best_score, best_config = 0.0, None
    # set scoring mode based on the number of classes
    average = 'binary' if np.unique(Y_dev).shape[0] == 2 else 'micro'

    print(f"Grid search over {len(params)} configs")
    print(f'Averaging: {average}')

    for i, config in enumerate(params):
        print(f'[{i}] Label Model')
        config = dict(zip(param_grid.keys(), config))
        # update default params if not specified
        config.update({
            param: value for param, value in defaults.items() \
            if param not in config})

        model = model_class(**model_class_init)
        # fit (estimate class balance with Y_dev)
        model.fit(L_train, Y_dev, **config)

        y_pred = model.predict(L_dev)
        y_gold = Y_dev

        # HACK = Snorkel 9.4 sometimes emits -1 predictions (why?)
        if -1 in y_pred:
            print("Label model predicted -1 (TODO: why?)")
            continue

        # only evaluate dev score
        mask = []
        for i in range(L_dev.shape[0]):
            if not np.all(L_dev[i] == -1):
                mask.append(i)

        mask = np.array(mask)
        metrics = compute_metrics(Y_dev[mask], model.predict(L_dev[mask]))

        msgs = []
        if not best_score or metrics[val_metric] > best_score[val_metric]:
            print(config)
            best_score = metrics
            best_config = config

            # mask uncovered data points
            mask = [i for i in range(L_train.shape[0]) \
                    if not np.all(L_train[i] == -1)]
            msgs.append(
                f'Coverage: {(len(mask) / L_train.shape[0] * 100):2.1f}%'
            )

            if Y_train is not None:
                # filter out candidate spans without gold labels
                y_mask = [i for i in range(len(Y_train)) if Y_train[i] != -1]
                mask = np.array(sorted(list(set(y_mask).intersection(mask))))
                metrics = compute_metrics(Y_train[mask],
                                          model.predict(L_train[mask]))
                msgs.append(
                    'TRAIN {}'.format(' | '.join(
                        [f'{m}: {v * 100:2.2f}' for m, v in metrics.items()])
                    )
                )

            msgs.append(
                'DEV   {}'.format(' | '.join(
                    [f'{m}: {v * 100:2.2f}' for m, v in best_score.items()]))
            )

        if verbose and msgs:
            print('\n'.join(msgs) + ('\n' + '-' * 80))

        if i % 50 == 0:
            print(f'[{i}] Label Model')

    # retrain best model
    if verbose:
        print('BEST')
        print(best_config)
    model = model_class(**model_class_init)
    model.fit(L_train, Y_dev, **best_config)
    return model, best_config


def grid_search(model_class,
                model_class_init,
                param_grid,
                train=None,
                dev=None,
                other_train=None,
                n_model_search=5,
                val_metric='f1',
                seed=1234,
                seq_eval=True,
                checkpoint_gt_mv=True,
                tag_fmt_ckpnt='BIO'):
    """Simple grid search helper function

    Parameters
    ----------
    model_class
    model_class_init
    param_grid
    train
    dev
    n_model_search
    val_metric
    seed
    seq_eval

    Returns
    -------

    """
    print(f"Using {'TOKEN' if not seq_eval else 'SEQUENCE'} dev checkpointing")
    if seq_eval:
        print(f"Using {tag_fmt_ckpnt} dev checkpointing")

    idx2tag = {0:'O', 1:'I-X', 2:'B-X'}

    L_train, Y_train, X_train_lens = train
    L_dev, Y_dev, X_dev_lens = dev

    # sample configs
    params = sample_param_grid(param_grid, seed)[:n_model_search]

    defaults = {'optimizer': 'adam'}
    best_score, best_config = 0.0, None
    print(f"Grid search over {len(params)} configs")

    for i, config in enumerate(params):
        print(f'[{i}] Label Model')
        config = dict(zip(param_grid.keys(), config))
        # update default params if not specified
        config.update({param: value for param, value in defaults.items() if param not in config})

        model = model_class(**model_class_init)
        # fit (estimate class balance with Y_dev)
        # HACK for BIO tag evaluation
        if len(np.unique(Y_dev)) != 2:
            Y_dev_hat = np.array([0 if y == 0 else 1 for y in Y_dev])
        else:
            Y_dev_hat = Y_dev
        model.fit(L_train, Y_dev_hat, **config)

        y_pred = model.predict(L_dev)

        # set gold tags for evaluation
        if tag_fmt_ckpnt == 'IO':
            y_gold = np.array([0 if y == 0 else 1 for y in Y_dev])
        else:
            y_gold = Y_dev

        if -1 in y_pred:
            print("Label model predicted -1 (TODO: this happens inconsistently)")
            continue

        # score on dev set (token or sequence-level)
        if seq_eval:
            metrics = score_sequences(*tokens_to_sequences(y_gold, y_pred, X_dev_lens, idx2tag=idx2tag))
        else:
            # HACK - use internal label model scorer
            metrics = model.score(L=L_dev,
                                  Y=y_gold,
                                  metrics=['accuracy', 'precision', 'recall', 'f1'],
                                  tie_break_policy=0)

        # compare learned model against MV on same labeled dev set
        # skip if LM less than MV
        if checkpoint_gt_mv:
            if seq_eval:
                mv_y_pred = mv(L_dev, 0)
                mv_metrics = score_sequences(
                    *tokens_to_sequences(y_gold, mv_y_pred, X_dev_lens, idx2tag=idx2tag)
                )
            else:
                metrics = model.score(L=L_dev,
                                      Y=y_gold,
                                      metrics=['accuracy', 'precision', 'recall', 'f1'],
                                      tie_break_policy=0)

            if metrics[val_metric] < mv_metrics[val_metric]:
                continue

        if not best_score or metrics[val_metric] > best_score[val_metric]:
            print(config)
            best_score = metrics
            best_config = config

            # print training set score if we have labeled data
            if np.any(Y_train):
                y_pred = model.predict(L_train)
                #y_gold = Y_train

                if tag_fmt_ckpnt == 'IO':
                    y_gold = np.array([0 if y == 0 else 1 for y in Y_train])
                else:
                    y_gold = Y_train

                if seq_eval:
                    metrics = score_sequences(*tokens_to_sequences(y_gold, y_pred, X_train_lens, idx2tag=idx2tag))
                else:
                    metrics = model.score(L=L_train,
                                          Y=y_gold,
                                          metrics=['accuracy', 'precision', 'recall', 'f1'],
                                          tie_break_policy=0)

                print('[TRAIN] {}'.format(' | '.join([f'{m}: {v * 100:2.2f}' for m, v in metrics.items()])))

            print('[DEV]   {}'.format(' | '.join([f'{m}: {v * 100:2.2f}' for m, v in best_score.items()])))
            print('-' * 88)

    # HACK - retrain best model
    print('BEST')
    print(best_config)
    model = model_class(**model_class_init)
    # HACK for BIO tag evaluation
    if len(np.unique(Y_dev)) != 2:
        Y_dev_hat = np.array([0 if y == 0 else 1 for y in Y_dev])
    else:
        Y_dev_hat = Y_dev
    model.fit(L_train, Y_dev_hat, **best_config)
    return model, best_config
