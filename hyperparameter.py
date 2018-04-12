class Hypermater():
    def __init__(self):
        self.train_path = "./smalldata/my_data_0/train_0.txt"
        self.dev_path = "./smalldata/my_data_0/dev_0.txt"
        self.test_path = "./smalldata/my_data_0/test_0.txt"

        self.embedding_path = "./word_embedding/glove.6B.100d.txt"
        self.save_words_embedding = "./word_embedding/words_embedding.pkl"
        self.save_labels_embedding = "./word_embedding/labels_embedding.pkl"

        self.embed_dim = 100
        self.hidden_size = 100
        self.num_layers = 1
        self.dropout = 0.4
        self.batch_size = 1
        self.lr = 0.001

        self.train_size = 0
        self.dev_size = 0
        self.test_size = 0

        self.word_num = 0
        self.label_num = 0

        self.train_print_acc = 10
        self.train_test = 10

        self.kernel_num = 100
        self.kernel_size = 3

        self.words_dict = None
        self.labels_dict = None

        self.use_crf = True
        self.use_cuda = False

        self.epochs = 5000
        self.proportional =True



