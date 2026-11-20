# PaperOrchestra M3 trial — 20 November 2026

User approved trying M3 with PaperOrchestra, then selected a local Gemini API configuration and confirmed saving the key. Draw only M3, not an entire paper. Keep all existing approved figure sources unchanged. No research model, scorer, Kaggle notebook, formal report or spreadsheet run/change.

Use the installed upstream `methods.agents.plotting_agent.process_single_figure` at commit d9bae997e793a015ed38bb1605be9eb58c00284f. Gemini model-list metadata check succeeded; `gemini-2.5-pro` and `gemini-3-pro-image` are listed. Select these explicitly rather than obsolete preview defaults. At most one critic round. Upstream PaperBanana examples are absent; the upstream retrieval function explicitly permits an empty-example fallback. This is therefore a no-few-shot PaperOrchestra trial, not a complete reference-augmented reproduction.

Mechanism verified against `baseline_final_pipeline/final_pipeline/xe/csa_patch_model.py` and `semantic_patch_model.py`, read as text only. Both object and relation pooling use the original object memory Vo; the relation attention query/residual uses Vr. Slots combine text, supported regional appearance and geometry. Spatial support plus validity masks prevent unsupported patch queries from receiving a prompt residual. No prompt-text or region encoder is trained by generating the drawing.

Deliver the actual raster output, generation description/caption/critic history and a receipt in a fresh Downloads directory. Inspect the actual image for swapped sources, missing residuals, reversed arrows, text errors and crowding before calling it usable. Raster output does not provide editable draw.io objects. Any vector reconstruction would be a separately disclosed derivative.

Amendment after first API attempt: despite appearing in model-list metadata, gemini-2.5-pro returned HTTP404 stating it is unavailable for new users and recommending gemini-3.1-pro-preview. No figure was produced by that attempt. Switch the planning/style/critic model to that explicit server-recommended name; keep the image model unchanged.

Second metadata correction: server-recommended gemini-3.1-pro-preview returned HTTP404 for v1/generateContent and is absent from the account's model listing. Use explicitly listed gemini-3.5-flash, which advertises generateContent. Keep both failed initial attempts recorded; neither produced an image.
