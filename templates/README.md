# Presentation & Visual Templates

This directory stores presentation templates, slide assets, and visual decks for the **MIS (Mechanistic Interpretability for SIR)** project.

---

## Files

* **[`Mechanistic_Interpretability_Templates.pptx`](Mechanistic_Interpretability_Templates.pptx)**: Master PowerPoint slide deck template for bi-weekly progress presentations, architecture breakdowns, and seminar reports at the Intuitive Robots Lab (KIT).
* **[`MIS_intro_presentation.pptx`](MIS_intro_presentation.pptx)**: Short project introduction (research question, SAE architecture, planned comparisons, D/R/C tests, questions for Paul).

Decks use the **Zilla Slab** font ([Google Fonts](https://fonts.google.com/specimen/Zilla+Slab)). Install it before presenting, or PowerPoint substitutes another font.

---

## Publishing to the website

Every `.pptx` in this folder **except the master template** appears automatically in the *Slide Sets* section of the [docs site](https://officiallygod.github.io/Mechanistic-Interpretability-for-SIR/#slides) on the next push to `main`. To add a new slide set:

1. Save the deck here as `<name>.pptx`.
2. Set its title and description in *File → Info → Properties* (Title, Comments). The website shows them; without a title the file name is used.
3. Push to `main`. The deploy workflow (`.github/scripts/build_slides_manifest.py`) copies the deck to the site and lists it, newest commit first.

---

## Linked Cloud Resources
* **Live Google Slides Deck:** [Google Slides Bi-Weekly Meetings](https://docs.google.com/presentation/d/1mFDo7Pkut4FRN9KIvGs0BECcrez3ElA9oxkdym2P1z8/edit?usp=sharing)
* **Miro Ideation Board:** [Miro Brainstorming Canvas](https://miro.com/welcomeonboard/Q3BMR0hTZ09UditVS2hqUU1zZGRBZ0RnTlVVRGc4OGtyYzZieUkycDkxakRtYm5YWnZSZ0lodTQ2N0o4Nys4bG5ra0xHMEMrSVpaVTFCVkNPYllQd21yRE5KcTdYZDZxNXNFdFF3Mnd1ZWVHV1VHSFdxN25DNEtCOW1uVXdjNnp3VHhHVHd5UWtSM1BidUtUYmxycDRnPT0hdjE=?share_link_id=911869718076)
