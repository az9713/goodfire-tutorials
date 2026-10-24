# Goodfire Tutorials

[Read the live tutorials](https://az9713.github.io/goodfire-tutorials/)

A complete, fixed self-study collection for understanding the neural-network
interpretability research published by [Goodfire](https://www.goodfire.com/).
The tutorials explain the arguments, mathematical ideas, evidence, limitations,
and experiments behind the essays. This is an independent educational project,
not an official Goodfire publication or an endorsement by its researchers.

## What is included

- **56 source companions**, with extended treatment of the original five essays.
- **Six optional preparation chapters** covering linear algebra, neural-network
  computation, causal interventions, statistics, geometry, and reproducible Python.
- **186 exercises with separate worked solutions**, available from the start.
- **13 runnable CPU experiments** with saved results and six explanatory figures.
- A recommended reading map, comparisons across methods, and three final
  investigations for applying the ideas to small neural networks.

The collection covers native neurons, sparse autoencoders, neural geometry,
circuits, parameter decomposition, training-data attribution, monitoring,
biological models, and research practice. The homepage and geometry-series hub
are listed as navigation sources alongside the 56 substantive publications.

## Start reading

Open the [course index](https://az9713.github.io/goodfire-tutorials/learning/index.html)
and its [reading map](https://az9713.github.io/goodfire-tutorials/learning/reading-map.html).
For a direct entry into the central debate, read the companion to Nick
Cammarata's [Neural Networks Are Already Clean](https://az9713.github.io/goodfire-tutorials/learning/guides/s02.html),
alongside [the original Goodfire essay](https://already-clean.goodfire.com/).

All lessons and answers are accessible immediately. There is no assessment gate,
account requirement, progress tracking, analytics, or hosted-model dependency.
The public site links to Goodfire and other publishers for original sources;
the local source archive, full downloaded essays, and PDFs are not republished.

## Read offline or run the experiments

Clone or download the repository, then open `docs/index.html` in a browser.
The tutorials, equations, figures, and solutions work offline because MathJax
and the other site assets are included. Reading the linked original publications
requires an internet connection or separately saved copies.

The experiments use Python, NumPy, PyTorch, and Matplotlib. A recorded reference
environment is available in `docs/learning/evidence/runtime-inventory.json`.
Install compatible packages before going offline, then run from the repository root:

```powershell
python docs/learning/labs/run_lab.py --lab all
python docs/learning/labs/run_lab.py --lab geometry
```

Each experiment writes a JSON receipt; the two trained models also save small
state dictionaries. They use generated data and CPU execution, with no downloads,
API keys, or paid services. These are teaching experiments on small constructions,
not reproductions of the publications' full-scale model results.

## Repository and deployment

`docs/` contains the complete static website. GitHub Pages publishes the `/docs`
directory on the `main` branch; `.nojekyll` preserves the static file layout.
The HTML files render as normal web pages at the live site rather than as GitHub
source previews. No server-side build is required.

Run the privacy and local-link checks with:

```powershell
python scripts/verify_publication.py
```

## Evidence and limitations

The teaching material was produced with AI assistance and checked through separate
scientific review, source comparisons, mathematical and numerical checks, and
browser validation. The live site's [coverage page](https://az9713.github.io/goodfire-tutorials/learning/coverage.html)
describes its scope. Research claims remain attributed to their sources; the
course distinguishes reported results, teaching derivations, measured local
experiments, and proposed replications.

Original lecture videos, external model weights, live APIs, and every publisher
interaction are outside this repository. Completing the readings does not certify
equivalence to any researcher's expertise. See [NOTICE.md](NOTICE.md) for source
attribution and third-party notices and [PRIVACY.md](PRIVACY.md) for the publication boundary.
