#pragma once

#include <cstdint>

// LeanKV Phase 7a: Embedded multi-domain calibration corpus.
//
// Used for first-load codebook calibration when no cached codebook exists
// for the loaded model. Shipping the text in the binary means the calibration
// flow has no external-file dependency — the same corpus is used for every
// model on every machine, making fingerprints stable and reproducible.
//
// Source: /tmp/calib_corpus.txt (checked in to LeanKV docs/phase7-calibration/).
// Contents (442 words, ~651 tokens on a typical BPE tokenizer):
//   - ML / technical prose
//   - English pangrams
//   - Python code (fibonacci, transformers imports)
//   - Physics prose (general relativity)
//   - SQL query
//   - Sherlock Holmes dialogue
//   - Science fiction narrative
//   - Cell biology prose
//   - Multilingual phrases (English/French/German/Spanish/Chinese/Japanese)
//   - Gettysburg Address fragment
//   - Weather forecast
//
// The diversity matters more than the absolute length: Lloyd-Max levels
// converge quickly as long as the distribution is sufficiently varied
// across tokens. Empirical check: 75-token prompt vs 651-token corpus
// gave identical levels to within 1e-4 on Qwen3-4B.

namespace leankv {

inline constexpr const char * calib_corpus =
    "Machine learning models often suffer from overfitting when the training data is limited. "
    "Regularization techniques like dropout and weight decay help mitigate this issue. "
    "Another approach is data augmentation, which artificially expands the dataset.\n"
    "\n"
    "The quick brown fox jumps over the lazy dog. She sells seashells by the seashore. "
    "Peter Piper picked a peck of pickled peppers.\n"
    "\n"
    "def fibonacci(n):\n"
    "    if n <= 1:\n"
    "        return n\n"
    "    a, b = 0, 1\n"
    "    for _ in range(n - 1):\n"
    "        a, b = b, a + b\n"
    "    return b\n"
    "\n"
    "The theory of general relativity, proposed by Albert Einstein in 1915, describes gravity "
    "as a curvature of spacetime caused by mass and energy. Unlike Newtonian gravity, which "
    "treats gravity as a force acting instantaneously at a distance, general relativity predicts "
    "that massive objects warp the fabric of spacetime, and other objects follow geodesic paths "
    "through this curved geometry.\n"
    "\n"
    "SELECT u.name, COUNT(o.id) AS order_count FROM users u LEFT JOIN orders o ON u.id = o.user_id "
    "WHERE u.created_at > '2024-01-01' GROUP BY u.id ORDER BY order_count DESC LIMIT 10;\n"
    "\n"
    "\"Elementary, my dear Watson,\" said Holmes, lighting his pipe by the fire. \"The matter is "
    "quite simple once you observe the details. The mud on his boots, the callus on his finger, "
    "the slight asymmetry of his gait \xE2\x80\x94 each one tells a story.\"\n"
    "\n"
    "In the year 2157, the colony ship Artemis completed its journey to Proxima Centauri b. "
    "The crew awoke from cryosleep to find that the planet was not quite what the probes had "
    "suggested. Where there should have been barren rock, green forests stretched to the horizon.\n"
    "\n"
    "The mitochondrion is the powerhouse of the cell, generating adenosine triphosphate through "
    "oxidative phosphorylation. These organelles have their own DNA, a vestige of their bacterial "
    "origins from billions of years ago during a symbiotic event.\n"
    "\n"
    "import numpy as np\n"
    "from transformers import AutoModel, AutoTokenizer\n"
    "model = AutoModel.from_pretrained(\"bert-base-uncased\")\n"
    "tokenizer = AutoTokenizer.from_pretrained(\"bert-base-uncased\")\n"
    "inputs = tokenizer(\"Hello world\", return_tensors=\"pt\")\n"
    "outputs = model(**inputs)\n"
    "\n"
    "Le chat est sur la table. The cat is on the table. Die Katze sitzt auf dem Tisch. "
    "El gato est\xC3\xA1 en la mesa. \xE7\x8C\xAB\xE5\x9C\xA8\xE6\xA1\x8C\xE5\xAD\x90\xE4\xB8\x8A. "
    "\xE3\x81\xAD\xE3\x81\x93\xE3\x81\xAF\xE3\x83\x86\xE3\x83\xBC\xE3\x83\x96\xE3\x83\xAB"
    "\xE3\x81\xAE\xE4\xB8\x8A\xE3\x81\xAB\xE3\x81\x84\xE3\x81\xBE\xE3\x81\x99\xE3\x80\x82\n"
    "\n"
    "Our fathers brought forth on this continent a new nation, conceived in liberty, and dedicated "
    "to the proposition that all men are created equal. Now we are engaged in a great civil war, "
    "testing whether that nation, or any nation so conceived and so dedicated, can long endure.\n"
    "\n"
    "The weather forecast for tomorrow calls for partly cloudy skies with a high of 72 degrees "
    "Fahrenheit and a low of 55. Winds will be light and variable, shifting to the southeast in "
    "the afternoon. There is a 20 percent chance of isolated thunderstorms in the late evening.\n";

// Version tag — bump if the corpus content changes. Cached codebooks
// include this in the fingerprint so old caches are invalidated when
// we change the calibration input.
inline constexpr uint32_t calib_corpus_version = 1;

} // namespace leankv
