#coding=utf8
import sys, os, time, gc, json
from datetime import datetime
from torch.optim import Adam

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

from utils.args import init_args
from utils.initialization import *
from utils.example import Example
from utils.batch import from_example_list
from utils.vocab import PAD
from utils.logger import Logger
from model.slu_baseline_tagging import SLUTagging

# initialization params, output path, logger, random seed and torch.device
args = init_args(sys.argv[1:])
#set_random_seed(args.seed)
args.seed = eval(args.seed)
device = set_torch_device(args.device)
print("Initialization finished ...")
# print("Random seed is set to %d" % (args.seed))
print("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")

start_time = time.time()
train_path = os.path.join(args.dataroot, 'train.json')
dev_path = os.path.join(args.dataroot, 'development.json')
Example.configuration(args.dataroot, train_path=train_path, word2vec_path=args.word2vec_path)
train_dataset = Example.load_dataset(train_path)
dev_dataset = Example.load_dataset(dev_path)
print("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
print("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)))

args.vocab_size = Example.word_vocab.vocab_size
args.pad_idx = Example.word_vocab[PAD]
args.num_tags = Example.label_vocab.num_tags
args.tag_pad_idx = Example.label_vocab.convert_tag_to_idx(PAD)

model = parse_method(args, device)
Example.word2vec.load_embeddings(model.word_embed, Example.word_vocab, device=device)

# log result
datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
experiment_name = f'baseline.lr_{args.lr}.encoder_cell_{args.encoder_cell}.dropout_{args.dropout}.embed_{args.embed_size}.hidden_{args.hidden_size}.layer_{args.num_layer}.batch_{args.batch_size}.{datetime_now}'
exp_dir = os.path.join('result/', experiment_name)
os.makedirs(exp_dir, exist_ok=True)
if args.testing:
    logger = Logger.init_logger(filename=exp_dir + '/test.log')
else:
    logger = Logger.init_logger(filename=exp_dir + '/train.log')
args_print(args, logger)

if args.testing:
    check_point = torch.load(open('model.bin','rb'),map_location=device)
    model.load_state_dict(check_point['model'])
    print('Load saved model from root path')

def set_optimizer(model, args):
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    grouped_params = [{'params': list(set([p for n, p in params]))}]
    optimizer = Adam(grouped_params, lr=args.lr)
    return optimizer

def decode(choice):
    assert choice in ['train', 'dev']
    model.eval()
    dataset = train_dataset if choice == 'train' else dev_dataset
    predictions, labels = [], []
    total_loss, count = 0, 0
    with torch.no_grad():
        for i in range(0, len(dataset), args.batch_size):
            cur_dataset = dataset[i: i + args.batch_size]
            current_batch = from_example_list(args, cur_dataset, device, train=True)
            pred, label, loss = model.decode(Example.label_vocab, current_batch)
            for j in range(len(current_batch)):
                if any([l.split('-')[-1] not in current_batch.utt[j] for l in pred[j]]):
                    print(current_batch.utt[j], pred[j], label[j])
            predictions.extend(pred)
            labels.extend(label)
            total_loss += loss
            count += 1
        metrics = Example.evaluator.acc(predictions, labels)
    torch.cuda.empty_cache()
    gc.collect()
    return metrics, total_loss / count

def predict():
    model.eval()
    test_path = os.path.join(args.dataroot, 'test_unlabelled.json')
    test_dataset = Example.load_dataset(test_path)
    predictions = {}
    with torch.no_grad():
        for i in range(0, len(test_dataset), args.batch_size):
            cur_dataset = test_dataset[i: i + args.batch_size]
            current_batch = from_example_list(args, cur_dataset, device, train=False)
            pred = model.decode(Example.label_vocab, current_batch)
            for pi, p in enumerate(pred):
                did = current_batch.did[pi]
                predictions[did] = p
    test_json = json.load(open(test_path, 'r', encoding='utf-8'))
    ptr = 0
    for ei, example in enumerate(test_json):
        for ui, utt in enumerate(example):
            utt['pred'] = [pred.split('-') for pred in predictions[f"{ei}-{ui}"]]
            ptr += 1
    json.dump(test_json, open(os.path.join(args.dataroot, 'prediction.json'), 'w', encoding='utf-8'), indent=4, ensure_ascii=False)


if not args.testing:
    num_training_steps = ((len(train_dataset) + args.batch_size - 1) // args.batch_size) * args.max_epoch
    logger.info('Total training steps: %d' % (num_training_steps))
    all_result = {'acc':[], 'precision':[], 'recall':[], 'fscore':[]}
    for run in range(args.runs):
        logger.info(f'==================Run {run+1} begins, seed {args.seed[run]}==================')
        set_random_seed(args.seed[run])
        model.reset_parameters()

        optimizer = set_optimizer(model, args)
        nsamples, best_result = len(train_dataset), {'dev_acc': 0., 'dev_f1': 0.}
        train_index, step_size = np.arange(nsamples), args.batch_size
        logger.info('Start training ......')
        for i in range(args.max_epoch):
            start_time = time.time()
            epoch_loss = 0
            np.random.shuffle(train_index)
            model.train()
            count = 0
            for j in range(0, nsamples, step_size):
                cur_dataset = [train_dataset[k] for k in train_index[j: j + step_size]]
                current_batch = from_example_list(args, cur_dataset, device, train=True)
                output, loss = model(current_batch)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                count += 1
            #logger.info('Training: \tEpoch: %d\tTime: %.4f\tTraining Loss: %.4f' % (i, time.time() - start_time, epoch_loss / count))
            torch.cuda.empty_cache()
            gc.collect()

            start_time = time.time()
            metrics, dev_loss = decode('dev')
            dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
            logger.info('Epoch: %d\t %.4f  %.4f' % (i, epoch_loss / count, dev_loss))
            print('Evaluation: \tEpoch: %d\tTime: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, time.time() - start_time, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
            if dev_acc > best_result['dev_acc']:
                best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1'], best_result['iter'] = dev_loss, dev_acc, dev_fscore, i
                torch.save({
                    'epoch': i, 'model': model.state_dict(),
                    'optim': optimizer.state_dict(),
                }, open(f'model{run}.bin', 'wb', encoding='utf-8'))
                #logger.info('NEW BEST MODEL: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
        all_result['acc'].append(best_result['dev_acc'])
        all_result['precision'].append(best_result['dev_f1']['precision'])
        all_result['recall'].append(best_result['dev_f1']['recall'])
        all_result['fscore'].append(best_result['dev_f1']['fscore'])
        logger.info('FINAL BEST RESULT: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.4f\tDev fscore(p/r/f): (%.4f/%.4f/%.4f)' % (best_result['iter'], best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1']['precision'], best_result['dev_f1']['recall'], best_result['dev_f1']['fscore']))
    logger.info("Dev ACC:{:.2f}-+-{:.2f} Precision:{:.2f}-+-{:.2f} Recall:{:.2f}-+-{:.2f} F score:{:.2f}-+-{:.2f}".format(
        torch.tensor(all_result['acc']).mean(), torch.tensor(all_result['acc']).std(),
        torch.tensor(all_result['precision']).mean(), torch.tensor(all_result['precision']).std(),
        torch.tensor(all_result['recall']).mean(), torch.tensor(all_result['recall']).std(),
        torch.tensor(all_result['fscore']).mean(), torch.tensor(all_result['fscore']).std(),
    ))
else:
    set_random_seed(args.seed[1])
    start_time = time.time()
    metrics, dev_loss = decode('dev')
    dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
    predict()
    logger.info("Evaluation costs %.2fs ; Dev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)" % (time.time() - start_time, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
