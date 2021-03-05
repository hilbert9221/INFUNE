from time import strftime, gmtime

# name of the model
model = 'IF'
# ratio of training set
ratio = 0.8
options = 'structure profile content'
log = strftime('logs/{}_{}_{:.1f}_%m-%d_%H:%M:%S.txt'.format(
    model, ''.join([s[0] for s in options.split()]), ratio), gmtime())

supervised = True
# dimension of the embeddings
dim = 2 ** 8
# number of negative samples
neg = 5
# number of candidates
k = 30
cuda = 0
# sample user pairs with top (1 - percent)% similarities
percent = 99
epochs = 120
batch_size = 2 ** 7
lr = 5e-4
# param for early stop
stop = 3


def init_args(args):
    global cuda, model, ratio, options, log, epochs

    cuda = args.cuda
    model = args.model
    ratio = args.ratio
    if hasattr(args, 'options'):
        options = args.options
    epochs = args.epochs
    log = strftime('logs/{}_{}_{:.1f}_%m-%d_%H:%M:%S.txt'.format(
        model, ''.join([s[0] for s in options.split()]), ratio), gmtime())
