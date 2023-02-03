# Config file for HOPN

import os
import numpy as np
import random
import torch
import dgl


def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999
    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)
    os.environ["OMP_NUM_THREADS"] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)



SEED = 42
set_random_seed(SEED)


if torch.cuda.is_available():
#     DEVICE = torch.device("cuda:0")
    DEVICE = torch.device("cuda:1")
    print("Using GPU: ", DEVICE)
else:
    DEVICE = torch.device("cpu")
    print("Using CPU: ", DEVICE)


TOPIC_INFO_DICT = {
    'Northeast Delhi Riots 2020': "Why were there riots in Delhi 2020? The violence in the city's northeast erupted following nationwide protests against a controversial citizenship law Prime Minister Narendra Modi's government had passed in late 2019. On the eve of February 23, riots broke out in Northeast Delhi between Anti-Citizenship Amendment Act (CAA) and pro-CAA protestors. The violence took a communal turn and led to the death of over 53 people over the course of the next 10 days. More than 200 were left injured.",
   
    'Demonetization': "Is demonetization in India a failure? It had severe economic consequences for the economy. The currency notes that were demonetized amounted to nearly 85% of the economy's total cash; and 85% of total cash being immobilized suddenly had effects that were crippling in both the short and the long-term.",
   
    'Delhi Riots 2020': "Who was involved in the Delhi riots? The 2020 Delhi riots, or North East Delhi riots, were multiple waves of bloodshed, property destruction, and rioting in North East Delhi, beginning on 23 February 2020 and caused chiefly by Hindu mobs attacking Muslims. Muslims were marked as targets for violence.",
   
    'Brexit': "What are the causes of Brexit? Factors include sovereignty, immigration, the economy, and anti-establishment politics, amongst various other influences. According to a December 2017 Financial Times analysis, the Brexit referendum results had reduced national British income by 0.6% and 1.3%.",
   
    'Never Trump Campaign': "Why do so many people support Trump? Men and women who endorse 'hegemonic masculinity' -- an idealized form of manhood where White, heterosexual men have power, status and dominance over women, gay men, men with disabilities, racial or religious minorities, and other groups -- are more likely to be Donald Trump's supporters. Trump reminds us of past dictators who have manipulated the masses using intimidation, lies, and propaganda to cause discontent, chaos, civil unrest, violence, and genocide.",
   
    '#HinduLivesMatter': "Are Hindus discriminated in western world history? Hindus have experienced both historical and ongoing religious persecution and systematic violence, in the form of forced conversions, documented massacres, demolition and desecration of temples, as well as the destruction of educational centers.",
   
    'Umar Khalid JNU': "Who is Umar Khalid? Umar Khalid is an Indian activist, a former student of Jawaharlal Nehru University, and a former leader of the Democratic Students' Union (DSU) in JNU. On September 14th, 2020, Khalid was arrested by the Delhi Police Special Cell as an alleged conspirator in the Delhi Riots case. He was charged with allegedly damaging public property, committing unlawful activities, raising funds, sedition, murder, attempt to murder, promoting enmity between Hindus and Muslims."
}


ID2LABEL = {
    0: 'hate',
    1: 'offensive',
    2: 'provocative',
    3: 'none-hate'
}


INPUT_PATH = ''
COLUMN = 'clean_text'
LABEL = 'label'

EXEMPLAR_PATH = ''
EXEMPLAR_EMBEDDING_PATH = ''
TIMELINE_PATH = ''
TIMELINE_EMBEDDING_PATH = ''
PROMPT_FILES = []
PROMPT_EMBEDDING_PATH = ''


TWEET_MAX_LEN = 300
SEQ_DIM = 128

GRAPH_SEQ_DIM = 32
EXEMPLAR_SEQ_DIM = 5
TIMELINE_SEQ_DIM = 20
GRAPH_DIM = 128
NUM_HEADS = 8

BATCH_SIZE = 32
MAX_EPOCHS = 20

USE_GRAPH = True
USE_GRAPH_EDGE = True
USE_USER_EMBEDDING = True

USE_TIMELINE = True
USE_EXEMPLAR = True

USE_EVIDENCE = False
USE_EVIDENCE_EARLY = False
USE_FAQ =  False

MODEL_TYPE = 'fusion'
# MODEL_TYPE = 'dml'

DML_MIXER = False

FUSION_LAYER = 11
# FUSION_TYPE = 'late'
# FUSION_TYPE = 'simple'
FUSION_TYPE = 'attention'
# FUSION_TYPE = 'stacked-attention'

DROPOUT_RATE = 0.25
# BASE_LEARNING_RATE = 1e-06
# CLASSIFIER_LEARNING_RATE = 1e-06
BASE_LEARNING_RATE = 4e-05
CLASSIFIER_LEARNING_RATE = 4e-05
# CLASSIFIER_LEARNING_RATE = 1e-04
WEIGHT_DECAY = 1e-4

KL_LOSS_LAMBDA = 1
LOGITS_COMBINATION_TYPE = 'average'
# LOGITS_COMBINATION_TYPE= 'confidence'
SAVE_RESULTS = True

print("\nTWEET_MAX_LEN: ", TWEET_MAX_LEN)
print("GRAPH_SEQ_DIM: ", GRAPH_SEQ_DIM)
print("EXEMPLAR_SEQ_DIM: ", EXEMPLAR_SEQ_DIM)
print("TIMELINE_SEQ_DIM: ", TIMELINE_SEQ_DIM)
print("SEQ_DIM: ", SEQ_DIM)
print("GRAPH_DIM: ", GRAPH_DIM)
print("NUM_HEADS: ", NUM_HEADS)
print("\nBATCH_SIZE: ", BATCH_SIZE)
print("MAX_EPOCHS: ", MAX_EPOCHS)
print("DROPOUT_RATE: ", DROPOUT_RATE)
print("BASE_LEARNING_RATE: ", BASE_LEARNING_RATE)
print("CLASSIFIER_LEARNING_RATE: ", CLASSIFIER_LEARNING_RATE)
print("WEIGHT_DECAY: ", WEIGHT_DECAY)
print("\nUSE_GRAPH: ", USE_GRAPH)
print("USE_GRAPH_EDGE: ", USE_GRAPH_EDGE)
print("USE_EXEMPLAR: ", USE_EXEMPLAR)
print("USE_TIMELINE: ", USE_TIMELINE)
print("USE_EVIDENCE: ", USE_EVIDENCE)
print("USE_EVIDENCE_EARLY: ", USE_EVIDENCE_EARLY)
print("USE_FAQ: ", USE_FAQ)
print("\nFUSION_LAYER: ", FUSION_LAYER)
print("MODEL_TYPE: ", MODEL_TYPE)
print("FUSION_TYPE: ", FUSION_TYPE)
print("KL_LOSS_LAMBDA: ", KL_LOSS_LAMBDA)
print("LOGITS_COMBINATION_TYPE: ", LOGITS_COMBINATION_TYPE)
print("SAVE_RESULTS: ", SAVE_RESULTS)
print("USE_CUSTOM_MODEL: ", USE_CUSTOM_MODEL)
print("CUSTOM_MODEL: ", CUSTOM_MODEL)
