# MOTIVE: method, GNN mechanics, and the run we did

A self-contained explainer: what the paper proposes, how the GNN actually computes things (matrices and formulas), what differs between training and inference, the role of Cell Painting node features, what happens to nodes unseen during training, and a summary of the sweep we ran.

## 1. What the paper presents

MOTIVE is a dataset + benchmark for **drug-target interaction (DTI) prediction** posed as **link prediction on a graph**.
Nodes are compounds (sources) and genes (targets); edges are known interactions.
The novel ingredient is the **node features**: instead of molecular/protein structure, every node is represented by its **Cell Painting morphological profile** from JUMP - what a cell looks like after that compound or gene perturbation.
Compounds get a 737-dim vector, genes a 722-dim vector.

The graph has three edge types: compound-gene (source-target, the thing predicted), compound-compound (source-source), gene-gene (target-target), assembled from seven databases (~11.5k genes, ~3.6k compounds, ~303k interactions).
Four graph variants (`bipartite`, `s_expanded`, `t_expanded`, `st_expanded`) let you measure what each edge type adds.

Three evaluation splits make the benchmark rigorous: **random** (transductive), **cold-source** (held-out new drugs), and **cold-target** (held-out new genes).
The cold splits are inductive - the held-out nodes are completely isolated at test time, so the model must rely on node features.

Headline result: GNNs that use Cell Painting features beat structure-only GNNs, feature-only models (MLP/Bilinear), and topological heuristics (shortest path) - and the advantage holds in the hard inductive settings.
GIN is the strongest convolution.

## 2. How the GNN works (GraphSAGE_CP)

Let `h` = hidden width (256 in the paper, 1024 in `gnn.json`).

**Fixed inputs (never updated):**

- `X_s` shape `[3632, 737]` - one frozen Cell Painting vector per compound.
- `X_t` shape `[4505, 722]` - one per gene.
- the graph (edge list). Structure is given, not learned.

**Learnable parameters - the whole model is a handful of matrices:**

| matrix | shape | role |
|---|---|---|
| `E_s` | `737 x h` | project compound features into a shared space |
| `E_t` | `722 x h` | project gene features into a shared space |
| `W_self^1`, `W_nbr^1` | `h x h` each | layer-1 conv: self term + neighbor term |
| `W_self^2`, `W_nbr^2` | `h x h` each | layer-2 conv |

(plus bias vectors). The classifier is a plain dot product - **zero parameters**.

Crucially these matrices are **fixed-size and independent of the number of nodes**.
The model is a *function* that maps (a node's features + its neighborhood) to an embedding, not a per-node lookup table.

## 3. The forward pass

For a compound `s` and gene `t`:

```
# Step 0 - project both node types into one h-dim space
h_s^0 = ReLU( x_s . E_s )          # x_s = frozen CP vector
h_t^0 = ReLU( x_t . E_t )

# Step 1 - message-passing layer 1 (per node v)
agg_v = mean over u in Neighbors(v) of  h_u^0
h_v^1 = LeakyReLU( W_self^1 . h_v^0  +  W_nbr^1 . agg_v )   # then L2-normalize

# Step 2 - layer 2 (now each node sees 2-hop neighbors)
h_v^2 = normalize( W_self^2 . h_v^1  +  W_nbr^2 . mean_u(h_u^1) )

# Step 3 - skip connection
z_v = h_v^1 + h_v^2

# Step 4 - score a candidate edge
score(s, t) = sigmoid( z_s . z_t )
```

The two `W` matrices per layer decide how much of "yourself" vs "your neighborhood" each embedding keeps.
Two layers means a 2-hop receptive field; the skip connection re-injects the 1-hop signal.
(Because there are several edge types, PyG's `to_hetero` clones `W_nbr` per relation - `binds`, its reverse, `similar` - and sums their messages.)

This is the standard **encoder-decoder** link-prediction recipe: a GNN encoder produces `z`, an inner-product decoder scores pairs.
What it learns geometrically: an embedding space where the dot product is high for interacting pairs and low for non-interacting ones - metric learning over the graph.
(It is dot product, not cosine - magnitude survives, which lets embedding norm encode hubness/degree.)

## 4. Training vs. inference

**Training.**
Take a batch of true compound-gene edges plus sampled non-edges, run Steps 0-4, compare `score` to the 1/0 label, and backprop.
The loss is BCE (paper) or a weighted BPR ranking loss (motivev2).
Gradients update **only `E_s, E_t, W^1, W^2`** (via Adam).
The Cell Painting features `X` and the graph stay frozen.
Model selection: best validation Hits@500 (paper) or minimum validation loss (motivev2).

A methods detail that matters: edges are split into **message** edges (fed to the GNN as structure for aggregation) and **supervision** edges (the labels being scored).
The edges you predict are never given to the GNN as input - that would be leakage.
Negative sampling is custom-built to never sample a held-out cold node into training negatives.

**Inference.**
The trained `E`/`W` matrices *are* the model.
Run the identical forward pass with weights frozen, no backprop: feed node features + graph, get `z`, dot-product any compound-gene pair you want a score for.
Nothing else "remains" - the embeddings are recomputed on the fly from features + neighbors each time.

## 5. When node features (Cell Painting) are available

Step 0 turns the frozen CP vector into the node's starting embedding, and message passing refines it with neighbor information.
Because the model is a function of features, it generalizes: two nodes that have never been linked but *look alike* (similar morphology, similar neighbors) land close in the space and score highly - so the model predicts genuinely new edges rather than memorizing training links.
This is the whole reason features matter, and why they matter most when the graph is sparse (the ablation shows structure starts to dominate as the graph gets denser).

## 6. When a node was NOT seen in training (the inductive / cold case)

A held-out compound or gene has a Cell Painting vector but **no edges visible to the model**.

**Does it get any edges at inference time? No - by design.**
In the cold split, every edge that the held-out node participates in is in the *test* set (the thing being predicted), so none of them are available as message-passing structure.
The node is fully **isolated** during the forward pass.
Giving it any of its true edges would leak the labels and defeat the inductive test.
(The zero-shot probe in the paper is even stricter - *both* endpoints unseen.)

So its embedding collapses gracefully to features only:

```
agg_v = mean over (no neighbors) = empty
z_v   driven entirely by  W_self . (its projected CP features)
```

The candidate links are still scored as dot products `z_s . z_t`, but the GNN never consumes those candidate edges as input - it only ever produced `z_s` and `z_t` from features (and, for seen nodes, their neighborhoods).

**Contrast - the featureless model (`GraphSAGE_embs`).**
With no CP features you instead learn a `[num_nodes x h]` embedding table - one free vector per node, shaped purely by connectivity.
Its parameters are tied to specific node identities, so it is **transductive only**: a node unseen in training has no row to look up and no neighbors to aggregate, so it cannot be embedded at all.
That is exactly why the paper marks it N/A on the cold splits.
The tell: CP model parameters are independent of node count (a function -> inductive); embs parameters scale with node count (a lookup table -> transductive).

## 7. The run we did

Environment: motivev2 branch (pixi, conda-forge stack), trained on `spirit` (4x H100 NVL).
motivev2 differs from the paper: BPR loss + min-val-loss selection (not BCE + Hits@500), `gnn.json` HPs (hidden 1024, neg_ratio 100, 200 epochs), plus sampled + cartesian (all-pairs) inference and mAP / success@k metrics.

**Single validated run** - cold-target, bipartite, GraphSAGE_CP, 200 epochs (`cc46fa`), on the realistic `sampled` eval:

- source mAP = **0.418** (47% of compounds significant at p<0.05)
- target mAP = **0.194** (75% of genes significant)
- success@15 = **0.336** vs random 0.032 (~10x over random)

(The `cartesian`/all-pairs metrics are much lower - source mAP 0.014 - because scoring every compound x gene pair is far harsher. Expected.)

**ORF sweep** - 35/36 non-GAT configs across model x graph x split (gnn/gin/mlp + most bilinear; GAT skipped, one bilinear/st_expanded cell OOM'd at cartesian).
Qualitatively reproduces the paper: GIN wins consistently (e.g. source/st_expanded `gin:cp` mAP 0.776, success@15 0.907; random/st_expanded `gin:embs` mAP 0.799), everything beats random by 10-100x, and cold-target is the hardest split.

Caveats: every config used `gnn.json`'s GNN-tuned HPs, so these are **indicative, not paper-reproducing** (the paper tuned HPs per config); the `roc_auc`/`acc` columns are unreliable on the imbalanced cartesian set (trust mAP / success@k); and the `embs` numbers on cold splits should be distrusted (unseen nodes have untrained embedding rows).

**Saved results** (on spirit, under the repo):
```
outputs/orf/{split}/{graph}/{model}/{hash}/
  {sampled,cartesian}/test/results.parquet      # per-pair scores
  {sampled,cartesian}/test/metrics.parquet      # collated metrics
  {sampled,cartesian}/test/metrics/*.npy        # individual metrics (mAP, success@k, hits@500, ...)
  weights.pt                                     # trained model
```
The cold-target/bipartite/GraphSAGE_CP run is at `outputs/orf/target/bipartite/gnn/cc46fa/`.
