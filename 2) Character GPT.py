""" This educational material draws inspiration from Andrej Karpathy’s character-wise token GPT "Let's build GPT: from
scratch, in code, spelled out" (2022). Indicative variable names with detailed code comments are introduced to enhance
intuitive understanding. You can directly run this program to view the printed functioning of each component.

Note: Hyperparameters have been intentionally minimized for educational purposes, which may limit model performance. 
Please scale up key parameters (e.g., embedding dimensions, layer count, and network width) based on available
computational resources to achieve optimal results.

Development Credit: This program was developed by Yijiao Cao (Identifier: 曹一骄1989-ShenzhenMiddleSchool2008-
XidianUniv2012-StonyBrookUniv2015), with all rights reserved. The unique academic identifier follows standard scholarly
disambiguation practices to ensure accurate attribution and distinguish the author within professional communities.
For inquiries, please contact: yijcao@qq.com. """


import torch
import torch.nn as nn
from torch.nn import functional as fun

torch.manual_seed(1234)  #  fixed random seed for reproducibility.
torch.set_printoptions(precision=1)  #  Sets global display to 2 decimal places



class Hyp:  #  Hyperparameters
  def __init__(self):
    self.batSiz = 4  # batch-Size
    self.drpRat = 0.1  # dropout-Rate
    self.embDim = 20  # embedding-Dimension
    self.lowTriSiz = 5  # lower-Triangular-Size
    self.numBlc = 6  # number-of-Blocks
    self.numHea = 4  # number-of-Heads (divider of embDim)
    self.numOptItr = 3001  # number-of-Optimization-Iterations
    self.optItrTrcInt = 300  # optimization-Iteration's-Tracing-Interval
    self.optLrnRat = 3e-3  # optimation-Learning-Rate
    if self.embDim % self.numHea != 0:
      raise ValueError(f"embDim ({self.embDim}) must be divisible by numHea ({self.numHea})")
    self.heaSiz = self.embDim // self.numHea

hyp = Hyp()



print('\n\n\n\n01. Read Text\n')

with open(r'Materials/Shurangama Sutra, Dharma Lotus Sutra, and Flower Adornment Sutra.txt', 'r', encoding='utf-8') as file:
  text = file.read()  #  read the file contents
print('a) Length: ', len(text))

chars = sorted(list(set(text)))  #  get all characters without repetition
print('\nb) Vocabulary/Characters: ', ''.join(chars))

vocSiz = len(chars)  #  number_of_vocabulary tells how many unique tokens are in the text
print("\nc) Size of the vocabulary:", vocSiz)



print('\n\n\n\n02.Encoder & Decoder')

str_ind_tbl = { ch:i for i,ch in enumerate(chars) }  #  string-indices table
encode = lambda s: [str_ind_tbl[c] for c in s]  #  string  ->  indices
print('\n\na) Example. "hello world!" is encoded as\n\n', encode("hello world!"))

ind_to_str = { i:ch for i,ch in enumerate(chars) }  #  indices-string table
decode = lambda l: ''.join([ind_to_str[i] for i in l])  #  indices  ->  string
print('\n\nb) Example. [3, 5, 7] is decoded as\n\n', decode([3, 5, 7]))

tokens = torch.tensor(encode(text), dtype=torch.long)  #  characters  ->  indices
print('\n\nc) Tokenizing the first five characters of the text\n\n', text[:5], '  ->', tokens[:5])



print('\n\n\n\n03. Split: Training-Data & Validation-Data\n')

splPrc = int(0.9 * len(tokens))  #  split percentage = 0.9 means that 90% of tokens are used for training, and 10% for validation.
train_data = tokens[:splPrc]  #  training indices
print('a) Training indices:', train_data.shape, 'with type', train_data.dtype)

val_data = tokens[splPrc:]  #  validation indices
print('\nb) Validation indices:', val_data.shape, 'with type', val_data.dtype)



print('\n\n\n\n04. get-batch(): batched-Input-Token-Sequences & batched-Target-Token-Sequences')

def get_batch(split, batSiz, timSiz):  #  Get a batch of token sequences. (batSiz, timSiz)

  data = train_data if split == 'train' else val_data
  batStrTxtPos = torch.randint(len(data) - timSiz, (batSiz,))  #  batched-Starting-Text-Positions (batSiz)
  batInpTokSeq = torch.stack([data[i : i + timSiz] for i in batStrTxtPos])  #  batched-Input-Token-Sequences (batSiz, timSiz)  <-  batched-Starting-Text-Positions (batSiz)
  batTrgTokSeq = torch.stack([data[i + 1 : i + timSiz + 1] for i in batStrTxtPos])  #  batched-Target-Token-Sequences (batSiz, timSiz)  <-  batched-Starting-Text-Positions (batSiz)

  return batStrTxtPos, batInpTokSeq, batTrgTokSeq  #  Random starting positions, from which the input token sequences are expanded, along with their target token sequences.


with torch.no_grad():  #  Gradient tracking disabled for all operations within this block

  start_pos_1, batInpTokSeq_1, batTrgTokSeq_1 = get_batch(split='train', batSiz=3, timSiz=4)
  print('\n\na) Random starting positions  ->  along with their tokens\n\n', start_pos_1, '  ->', tokens[start_pos_1.tolist()],
        '\n\n\nb) Input and target tokens expanded from the starting positions ', '\n\n', batInpTokSeq_1, '\n\n', batTrgTokSeq_1)



print('\n\n\n\n05. Embedding: batched-Input-Token-Sequences  ->  batched-Input-Embeddings')


class Embedding(nn.Module):  #  batched-Input-Token-Sequences  ->  batched-Input-Embeddings
  def __init__(self, embDim, numPos): #  Create two embedding tables for token and position
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocSiz, embDim)  #  The tokEmbTbl contains vocSiz vectors. Each vector has dimension n_emb, to represent a token.
    self.position_embedding_table = nn.Embedding(numPos, embDim)  #  The position_embedding_table contains T_1 vectors. Each vector has dimension n_emb, to represent a position.

  def forward(self, batInpTokSeq):  #  batched-Input-Token-Sequences (B, T)  ->  batched-Input-Embeddings (B, T, embDim)
    B, T = batInpTokSeq.shape  #  batched-Token-Sequences (B, T)
    batPosEmb = self.position_embedding_table(torch.arange(T))  #  batched-Position-Embeddings (B, T, embDim)
    batInpTokEmb = self.token_embedding_table(batInpTokSeq)  #  batched-Input-Token-Embeddings (B, T, embDim)  <-  batched-Input-Token-Sequences (B, T)
    batInpEmb = batInpTokEmb + batPosEmb  #  batched-Input-Embeddings (B, T, embDim)  <-  batched-Input-Token-Embeddings (B, T, embDim) + batched-Position-Embeddings (B, T, embDim)
    return batInpEmb, batInpTokEmb, batPosEmb  #  batched-Input-Embeddings (B, T, embDim), batched-Input-Token-Embeddings (B, T, embDim), batched-Position-Embeddings (B, T, embDim)


with torch.no_grad():  #  Gradient tracking disabled for all operations within this block

  _, batInpTokSeq_2, _ = get_batch(split='train', batSiz=2, timSiz=3)
  print('\n\na) A batch of two token sequences from get_batch() \n\n', batInpTokSeq_2)

  embedding_1 = Embedding(embDim=4, numPos=3)  #  let numPos = batInpTokSeq_2.timSiz
  batInpEmb_1, batInpTokEmb_1, batPosEmb_1 = embedding_1(batInpTokSeq_2)
  print('\n\nb) Embedding for each token\n\n', batInpTokEmb_1,
        '\n\n\nc) Embedding for each position\n\n', batPosEmb_1,
        '\n\n\nd) Add the two modDic_1\n\n', batInpEmb_1)



print('\n\n\n\n06. Head(): Batched-Input-Embeddings  ->  Weighted-Sub-Mutual-Adjustments')

class Head(nn.Module):  #  Head attention provides raw material for MultiHead()'s adjusting each token embedding according to others.

  def __init__(self, embDim, heaSiz, lowTriSiz):  #  number-of-Embeddings, head-Size, lower-Triangular-Size

    super().__init__()
    self.heaSiz = heaSiz  #  Output dimension of raw material for embedding adjustment. A non-trainable constant stored in Head()
    self.query = nn.Linear(embDim, heaSiz, bias=False)  #  Asking: Who can get my attention?
    self.key = nn.Linear(embDim, heaSiz, bias=False)  #  Answering: Can I get your attention?
    self.value = nn.Linear(embDim, heaSiz, bias=False)  #  Raw material for token embedding adjustment if the answering closely meets the asking.
    self.dropout = nn.Dropout(hyp.drpRat)  #  Randomly zero out some elements to create a new pattern, avoiding over-dependence on some specific element, decreasing the risk of over-fitting.
    self.register_buffer('mask', torch.tril(torch.ones(lowTriSiz, lowTriSiz)))  #  A non-trainable lower-triangular-matrix tensor stored in Head()

  def forward(self, batTokEmb):  #  Batched-Input-Embeddings (B, T, embDim)  ->  Weighted-Sub-Mutual-Adjustments (B, T, h_size)

    B, T, _ = batTokEmb.shape  #  (B, T, embDim)
    queTst = self.query(batTokEmb)  #  question-Testing (B, T, heaSiz)
    queAns = self.key(batTokEmb)  #  question-Answering  (B, T, heaSiz)
    rawSubMutAdj = self.value(batTokEmb)  #  raw-Sub-Mutual-Adjustments (B, T, heaSiz)

    proWeiForMutAdj = queTst @ queAns.transpose(-1, -2) * (self.heaSiz ** -0.5)  #  pro-Weights-For-Mutual-Adjustments (B, T, T)
    bckLooProWeiForMutAdj = proWeiForMutAdj.masked_fill(self.mask[:T, :T] == 0, float('-inf'))  #  Back-Looking-Pro-Weights-For-Mutual-Adjustments (B, T, T)
    bckLooWeiForMutAdj = fun.softmax(bckLooProWeiForMutAdj, dim=-1)  #  Back-Looking-Weights-For-Mutual-Adjustments (B, T, T)
    psnBckLooWeiForMutAdj = self.dropout(bckLooWeiForMutAdj)  #  personalized-Back-Looking-Weights-For-Mutual-Adjustments (B, T, T) to avoid over-reliance on specific position
    weiSubMutAdj = psnBckLooWeiForMutAdj @ rawSubMutAdj  #  Weighted-Sub-Mutual-Adjustments (B, T, heaSiz)  <-  (B, T, T) * (B, T, heaSiz)  ->

    return weiSubMutAdj, rawSubMutAdj  #  Weighted-Sub-Mutual-Adjustments (B, T, heaSiz), raw-Sub-Mutual-Adjustments (B, T, heaSiz)


with torch.no_grad():  #  Gradient tracking disabled for all operations within this block

  _, batInpTokSeq_3, _ = get_batch(split='train', batSiz=2, timSiz=3)  #  (2, 3)
  print('\n\na) A batch of input tokens from get_batch()\n\n', batInpTokSeq_3)

  embedding_2 = Embedding(embDim=5, numPos=3)  #  Let the number-of-positions be timSiz
  batInpEmb_2, _, _ = embedding_2(batInpTokSeq_3)  #  (2, 3)  ->  (2, 3, 5)
  print('\n\nb) Embed the input tokens\n\n', batInpEmb_2)

  head_1 = Head(embDim=5, heaSiz=4, lowTriSiz=3)  #  lower-Triangular-Size = timSiz
  weiSubMutAdj_1, rawSubMutAdj_1 = head_1(batInpEmb_2)  #  Weights for embedding adjustment. Raw material for adjusting each token embedding according to others. (2, 3, 5)  ->  (2, 3, 4)
  print('\n\nc) weights-for-Sub-Mutual-Adjustment\n\n', weiSubMutAdj_1,
        '\n\n\nd) raw-Sub-Mutual-Adjustments (for modDic_1)\n\n', rawSubMutAdj_1)



print('\n\n\n\n07. MultiHead(): Multi-Head Attention Implementation')

class MultiHead(nn.Module):  #  Multi-head attention gives each token embedding adjustment according to others.

  def __init__(self, embDim, heaSiz, lowTriSiz):

    super().__init__()
    self.n_head = embDim // heaSiz
    self.heads = nn.ModuleList([Head(embDim=embDim, heaSiz=heaSiz, lowTriSiz=lowTriSiz) for _ in range(self.n_head)])  #
    self.adjPrj = nn.Linear(self.n_head * heaSiz, embDim)  #  adjustment-Projector()
    self.dropout = nn.Dropout(hyp.drpRat)  #  Dropout randomly personalizes a vector.

  def forward(self, batTokEmb):  #  batchedTokenEmbeddings (B, T, embDim)

    cctSubMutAdj = torch.cat([h(batTokEmb)[0] for h in self.heads], dim=-1)  #  concatenated-Sub-Mutual-Adjustments (B, T, n_head * heaSiz)  <-  batched-Token-Sequence (B, T, embDim)
    MutAdj = self.adjPrj(cctSubMutAdj)  #  MutuallyAdjustedEmbeddings (B, T, embDim)  <-  ConcatenatedMutuallyAdjustedSubEmbeddings (B, T, h_size) * n_head
    psnMutAdj = self.dropout(MutAdj)  #  personalized-Mutually-Adjustments (B, T, embDim)  <-  mutualAdjustments (B, T, embDim) for token modDic_1

    return self.n_head, cctSubMutAdj, psnMutAdj  #  concatenatedRawMaterials_for_tokenEmbeddingAdjustments (), mutualAdjustments_with_dropout (B, T, embDim)


with torch.no_grad():  #  Gradient tracking disabled for all operations within this block

  _, batInpTokSeq_4, _ = get_batch(split='train', batSiz=2, timSiz=3)  #  (2, 3)
  print('\n\na) A batch of input tokens from get_batch()\n\n', batInpTokSeq_4)

  embedding_3 = Embedding(embDim=7, numPos=4)  #  Let the number_of_positions be timSiz+1 for parametric distinction
  batInpEmb_3, _, _ = embedding_3(batInpTokSeq_3)  #  (2, 3)  ->  (2, 3, 7)
  print('\n\nb) Embed the input tokens\n\n', batInpEmb_3)

  head_2 = Head(embDim=7, heaSiz=3, lowTriSiz=3)
  _, raw_mat_2 = head_2(batInpEmb_3)
  print('\n\nc) weighted-Sub-Mutual-Adjustments  <-  Head( batched-Input-Embeddings )\n\n', raw_mat_2)  #  (B, T, n_head * heaSiz)

  multiHead_1 = MultiHead(embDim=7, heaSiz=3, lowTriSiz=3)
  nHead_1, conRawMat_tokEmbAdj_1, mutAdj_1 = multiHead_1(batInpEmb_3)  #  NumberOfHeads_1 (1), concatenatedRawMaterials_for_tokenEmbeddingAdjustments_1 (2, 3, n_head * heaSiz), mutual_adjustments_1 (2, 3, 7)
  print('\n\nd) personalized-Mutually-Adjustments by', nHead_1, 'Head()s\n\n', conRawMat_tokEmbAdj_1,  #  (B, T, n_head * heaSiz)
        '\n\n\ne) Mutual adjustment of token modDic_1 given by a linear layer on the concatenated raw materials\n\n', mutAdj_1)  #  (B, T, embDim)

  mutAdj_tokEmb_1 = batInpEmb_3 + mutAdj_1  #  mutuallyAdjusted_tokenEmbeddings_1 (2, 3, 7) = tokenEmbeddings_3 (2, 3, 7) + mutualAdjustments_1 (2, 3, 7)
  print('\n\nf) Adjusted Embeddings is the sum of Token modDic_1 (07.2.b) and their mutual adjustments (07.2.e).\n\n', mutAdj_tokEmb_1)  #  (2, 3, 7)



print('\n\n\n\n08. FFN(): Feed-Forward Network adjusts each token embedding within itself')

class FFN(nn.Module):  #  Feed-Forward Network
  def __init__(self, embDim):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(embDim, 4 * embDim),
      nn.ReLU(),
      nn.Linear(4 * embDim, embDim),
      nn.Dropout(hyp.drpRat),
    )

  def forward(self, x):  #  x is a batch of modDic_1 with shape (B, T, embDim)
    return self.net(x)  #  self.net(x) is the adjusted embedding within each token. (B, T, embDim)


with torch.no_grad():  #  Gradient tracking disabled for all operations within this block
  ffn_1 = FFN(7)  #  Set the dimension_of_embedding to that of mut_adj_1
  self_adj_1 = ffn_1(mutAdj_1)  #  Self-adjusted token modDic_1. (2, 3, 7)
  print('\n', self_adj_1)  #  (B, T, embDim)'



print('\n\n\n\n09. Block(): combining multi-head attention and position-wise feed-forward network.')

class Block(nn.Module):  #  Block() compounds the MultiHead()'s mutual embedding adjustment and the FFN()'s self embedding adjustment into one step.

  def __init__(self, embDim, heaSiz, lowTriSiz):  #  heaSiz

    super().__init__()
    self.multi_head = MultiHead(embDim=embDim, heaSiz=heaSiz, lowTriSiz=lowTriSiz)  #  self-attention
    self.ffn = FFN(embDim)
    self.lyrNrm_1 = nn.LayerNorm(embDim)  #  with learnable beta and gamma
    self.lyrNrm_2 = nn.LayerNorm(embDim)  #  with learnable beta and gamma

  def forward(self, OrgInpEmb):  #  Compound the MultiHead()'s mutual embedding adjustment and the FFN()'s self embedding adjustment. (B, T, n_emb)  ->  (B, T, n_emb)

    _, _, PsnMutAdjInpEmb = self.multi_head(self.lyrNrm_1(OrgInpEmb))  #  (B, T, n_emb)  ->  SequentiallyMutuallyAdjusted_tokenEmbeddings (B, T, n_emb)
    mutAdjTokEmb = OrgInpEmb + PsnMutAdjInpEmb  #  originalTokenEmbeddings (B, T, n_emb) + SequentiallyMutuallyAdjusted_tokenEmbeddings (B, T, n_emb)  ->
    slfMutAdjTokEmb = mutAdjTokEmb + self.ffn(self.lyrNrm_2(mutAdjTokEmb))  #  Adjust each token embedding within itself.

    return slfMutAdjTokEmb  #  Twice-adjusted modDic_1 of shape (B, T, n_emb)


with torch.no_grad():  #  Gradient tracking disabled for all operations within this block

  _, batInpTokSeq_5, _ = get_batch(split='train', batSiz=2, timSiz=3)  #  (2, 3)
  print('\n\na) A batch of input tokens from get_batch()\n\n', batInpTokSeq_5)

  embedding_4 = Embedding(embDim=7, numPos=4)  #  Let the number_of_positions be timSiz+1 for parametric distinction
  batInpEmb_4, _, _ = embedding_4(batInpTokSeq_5)  #  (2, 3)  ->  (2, 3, 7)
  print('\n\nb) Embed the input tokens\n\n', batInpEmb_4)

  block_1 = Block(7, 3, 3)
  blc_adj_1 = block_1(self_adj_1)  #  Block_Adjustment. (B, T, n_emb)  ->  (B, T, n_emb)
  print('\n\nc) Block() compounds mutual-adjustment and self-adjustment as one adjustment.\n\n', blc_adj_1)  #  (B, T, embDim)



print('\n\n\n\n10. Transformer(): Generative_Pre-trained_Transformer combines a Embedding() and several Block()s into a streamline')

class Transformer(nn.Module):

  def __init__(self, embDim, heaSiz, lowTriSiz, numBlc):  #  number_of_embeddings, heaSiz, number_of_positions, number_of_blocks
    super().__init__()
    self.lowTriSiz = lowTriSiz
    self.embedding = Embedding(embDim=embDim, numPos=lowTriSiz)  #
    self.seqBlc = nn.Sequential(*[Block(embDim, heaSiz, lowTriSiz) for _ in range(numBlc)])  #  The sequential_blocks compounds several block adjustments. Note: parameters of each block are independent, not repetition of one block.
    self.lyrNrm = nn.LayerNorm(embDim)   #  Added final layer norm
    self.vocPrj = nn.Linear(embDim, vocSiz)  #  vocabularyProjection: modDic_1 (B, T, embDim)  ->  logits (B, T, vocSiz) which are unsoftmaxed pro-probabilities to predict the next token

  def forward(self, batInpTokSeq, batTrgTokSeq = None):  #  input_tokens (B, T), target_tokens (B, T)  ->  logits (B, T, vocSiz), loss (1)
    B, T = batInpTokSeq.shape
    batInpEmb, _, _ = self.embedding(batInpTokSeq)  #  (B, T)  ->  (B, T, embDim)
    adjEmb = self.seqBlc(batInpEmb)   #  Original token modDic_1  ->  Adjusted modDic_1 of shape (B, T, embDim)
    adjEmb = self.lyrNrm(adjEmb)   #  Normalization of each embedding vector within itself
    logits = self.vocPrj(adjEmb)   #  LanguageModal projection: modDic_1 (B, T, embDim)  ->  logits (B, T, vocSiz) which are unsoftmaxed pro-probabilities for next-token-picking

    if batTrgTokSeq is None:
      loss = None
    else:
      cmbLgt = logits.view(B * T, vocSiz)  #  combined-logits (B * T, vocSiz)  <-  logits (B, T, vocSiz)
      cmbBatTrgTokSeq = batTrgTokSeq.view(B * T)  #  combined-Batched-Target-Token-Sequences (B * T)  <-  batTrgTokSeq (B, T)
      loss = fun.cross_entropy(cmbLgt, cmbBatTrgTokSeq)  #  average( cross-entropy( softmax(cmbLgt), oneHot(cmbBatTrgTokSeq) ) )

    return logits, loss  #  logits (B, T, vocSiz), loss (1)

  @torch.no_grad()  #  Turn off gradient tracking
  def generate(self, batExsTok, numNewTok):  #  batched-Existing-Tokens (B, T), number-of-New-Tokens (1)  ->  existing_tokens + generated_tokens (B, T + n_new_tok)
    for _ in range(numNewTok):  #  Iteration of updating existing_tokens by appending a new token each time
      batExsTokEnd = batExsTok[:, -self.lowTriSiz:]  #  batched-Existing-Tokens'-Endings (B, T)  <-  last_few_tokens (B, lt_siz)
      logits, _ = self(batExsTokEnd)  #  endings-of-Existing-Tokens (B, lowTriSiz)  ->  batched-Existing-Tokens'-Endings
      batLgtEnd = logits[:, -1, :]  #  batched-Logits'-Endings (B, vocSiz)  <-  (B, lowTriSiz, vocSiz)
      batPrbForNxtTok = fun.softmax(batLgtEnd, dim=-1)  #  batched-Probability-For-Next-Tokens (B, vocSiz)  ->  next_token's_probabilities (B, vocSiz).
      batNxtTok = torch.multinomial(batPrbForNxtTok, num_samples=1)  #  batched-Next-Tokens (B, vocSiz)  ->  next_token (B, 1)
      batExsTok = torch.cat((batExsTok, batNxtTok), dim=1)  #  existing_tokens (B, T), next_token (B, 1)  ->  existing_tokens (B, T + 1) ...
    return batExsTok  #  (B, T + n_new_tok)


with torch.no_grad():  #  Gradient tracking disabled for all operations within this block
  _, batInpTokSeq_6, batTrgTokSeq_2 = get_batch(split='train', batSiz=2, timSiz=4)  #  (2, 4)
  print('\n\na) Batched-Input-Token-Sequences from get_batch()\n\n', batInpTokSeq_6)

  gpt_1 = Transformer(embDim=6, heaSiz=3, lowTriSiz=4, numBlc=2)  #  let lower-Triangular-Size = timSiz
  logit_1, loss_1 = gpt_1(batInpTokSeq_6, batTrgTokSeq_2)  #  (2, 4), (2, 4)  ->  (2, 4, vocSiz), (1)
  print('\n\nb) Logits & Loss from Transformer()\n\n', logit_1, '\n\n', loss_1)


  strTok = torch.zeros((2, 1), dtype=torch.long)  #  a starting-Token (1, 1)
  genTokSeq = gpt_1.generate(strTok, 20)  #  generated-Token-Sequences (batch-Size: 2, time-Size: 1+20)
  exmGenTokSeq = genTokSeq[0].tolist()  #  example-Generated-Token-Sequence (21)
  print('\n\nc) Text Generation\n', decode(exmGenTokSeq))



print('\n\n\n\n11. Loss Estimation')

@torch.no_grad()  # Turns off gradient-tracking
def estimate_loss(model, dvs):  #  device
  evlItr = 200  # evaluation-Iteration
  avrLss = {}  # average-Losses of training data and validation data
  model.eval()  # evaluation mode disables dropout/batchNorm
  for split in ['train', 'value']:
    losses = torch.zeros(evlItr)  # losses container
    for k in range(evlItr):
      _, batInpTokSeq, batTrgTokSeq = get_batch(split=split, batSiz=hyp.batSiz, timSiz=hyp.lowTriSiz)
      batInpTokSeq = batInpTokSeq.to(device)
      batTrgTokSeq = batTrgTokSeq.to(device)
      _, loss = model(batInpTokSeq, batTrgTokSeq)
      losses[k] = loss.item()  #  fill in the losses container. item() extracts the scalar value from a single-element tensor
    avrLss[split] = losses.mean()
  model.train()  # re-enables dropout/batchNorm
  return avrLss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}\n")

# Initialize model and move to device
gpt_2 = Transformer(embDim=hyp.embDim, heaSiz=hyp.heaSiz, lowTriSiz=hyp.lowTriSiz, numBlc=hyp.numBlc).to(device)
avrLss_1 = estimate_loss(gpt_2, device)
print(avrLss_1)



print('\n\n\n\n12. Training')
optimizer = torch.optim.AdamW(gpt_2.parameters(), lr=hyp.optLrnRat)

print('\na) Text Generation Before Training:')
with torch.no_grad():
  exsTok = torch.zeros((1, 1), dtype=torch.long, device=device)  #  existing_tokens (1, 1)
  genTok = gpt_2.generate(batExsTok=exsTok, numNewTok=200)[0].tolist()  #  generated-tokens
print(decode(genTok))

print('\n\nb) Training:\n')
for itr in range(hyp.numOptItr):
  # Get batch and move to device
  _, batInpTokSeq_1, batTrgTokSeq_1 = get_batch(split='train', batSiz=hyp.batSiz, timSiz=hyp.lowTriSiz)
  batInpTokSeq_1 = batInpTokSeq_1.to(device)
  batTrgTokSeq_1 = batTrgTokSeq_1.to(device)
  # Training step
  logits_1, loss_1 = gpt_2(batInpTokSeq_1, batTrgTokSeq_1)
  optimizer.zero_grad(set_to_none=True)
  loss_1.backward()
  optimizer.step()
  # Periodic evaluation
  if itr % hyp.optItrTrcInt == 0:
    avrLss_2 = estimate_loss(gpt_2, device)
    print(f"step = {itr}: train loss {avrLss_2['train']:.4f}, value loss {avrLss_2['value']:.4f}")

print('\n\nc) Generation After Training:')
with torch.no_grad():
  exsTok = torch.zeros((1, 1), dtype=torch.long, device=device)
  genTok = gpt_2.generate(batExsTok=exsTok, numNewTok=200)[0].cpu().tolist()
print(decode(genTok), '\n')


