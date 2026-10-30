# Reading/delta notes cho shared-referent caption selection

30/10/2026. Companion của [proposal](SHARED_REFERENT_CAPTION_SELECTION_PROPOSAL_2026-10-30.md). Primary methods được dùng để chọn hướng, không phải danh sách “bài đã tìm” để tạo novelty. Không đọc lại ZIP hoặc5PDF của vòngpointer.

## Năm nearest mechanisms đã đọc

| Primary source | Read scope thực tế | Nearest delta / giới hạn novelty |
|---|---|---|
| [Johnsonetal., Image Retrieval using Scene Graphs, CVPR2015](https://openaccess.thecvf.com/content_cvpr_2015/papers/Johnson_Image_Retrieval_Using_2015_CVPR_paper.pdf) | §5.1–5.2Eq1–4; phầndataset§4.1–4.2 | **Đãcó đúng latentassignment dùngchung, unary/object-attribute factors và binaryrelation factors, MAPscore graph→image.** UnaryR-CNN, binaryGMMrelativegeometry. Proposedfactorization kếthừa; khác ở learnedobservedtruth/localcrops và captionpolicy trênnativeC0pool, khôngmộtgraphsolver mới. Không nói đổi chiềuretrieval tựnó tạo contribution. |
| [MAttNet,2018](https://arxiv.org/pdf/1801.08186) | §3.1,§3.2.3,§3.3Eq1 vàrankingloss; dataset/splitparagraph§4.1 | Subjectcandidate cốđịnh cùngmaxcontextrelation, weightedsubject/location/relationmodules. Singlepair/sharedsubjectgrounding khôngnew. Proposedwholecaptiongraph cóthểgiữmộtreferentqua≥2claims với nhiềucontextobjects, rồichọncảcaption; general structuredmatching vẫn đãcóJohnson. |
| [PICL, FindingsACL2023](https://aclanthology.org/2023.findings-acl.120.pdf) | §3.1–3.3Eq1–3 | Captioner+CLIPlistener đãrerankpartialcaptions/beam trongactualgeneration, normalizedovertarget/distractorimages. Proposedlistener là groundedstructuredwitness trongmộtảnh, wholecaptionpoolfixed; khôngclaimfirstlistenerdecoding. |
| [ComCLIP,NAACL2024](https://aclanthology.org/2024.naacl-long.370.pdf) | §4.2–4.4Eq4–5 vàimplementationparagraph | Subject/object/predicatecrops và textconceptweights đãcó; reweightedembeddingsgộpvớiglobalfeature. Proposedkhôngchỉcrop/gộptrọngsố: thamchiếuchung ràngbuộcassignment trongscorecácclaims. Khôngnhậncausalproof từcrop/tableintervention. |
| [CORA,2024](https://arxiv.org/pdf/2406.11820) | §3.1–3.2Eq1; §3.4Eq8–13; §4.4Eq14; §3.3chỉoverview, chưakiểmhếtGATequations | Captiongraphobject/attributes/relations, GATembedding; global/scene/entityalignment; **đãcó entity-min reranking**. Proposedkhôngnói mọiprior chỉbagofwords; giữ explicitROIassignment qua jointclaims thaypooledgraph/globalembedding. Utility/novelty của adaptationchưachứngminh. |

Đối chiếu5sourceskhôngchứngminh vắngmọiprior structuredcaptionreranking. Mứcđộmới dựkiến hẹp; nếu chỉ là đưaJohnson scorer vào captionpool mà không thay learning/usefulbehavior vượtmatchedbaseline, khôngnhận contribution phươngpháp.

## Đọc cho hai hướng không chọn

- [COMM / From CLIP to DINO](https://arxiv.org/pdf/2310.08825): §4Eq1, featurefusion vàTrainingDetails. Multi-levelCLIP/DINO fusion+coadaptlanguage đãcó. Chưakiểmvenueaccepted của bảnPDF, khônggán từtrí nhớ. KhôngdùngreportedlargeGPUcost làmETA của pilotnhỏ.
- [MOCHa,EMNLP2024](https://aclanthology.org/2024.emnlp-main.1263.pdf): §3.1–3.2. Fidelity từreferenceNLIcontradiction, adequacyBERTScore, KL/PPO. Image-groundedclaimreward là thayđổi semantic supervision cóthểnghiêncứu nhưngcầnverifiertrước; chưachọnchạyRL.
- [LURE,ICLR2024](https://arxiv.org/pdf/2310.00754): §3.1Alg1,§3.2deploymentparagraph. Wholecaptionrevisor/GPTcorruptions đãcó; khôngphảinewselectedmethod. Khôngmượnnhãnsyntheticabsence của nó làmverifiednegative cho proposal.

## Nguồn supervision / stresscheck

- [GQA,CVPR2019](https://arxiv.org/pdf/1902.09506): §3.1SceneGraphNormalization, splitparagraph. Scenegraphs gốcVG113kCOCO/Flickrimages, cóobjectIDs/boxes/attributes/relations vànormalizedontology. Dữliệuđãnormalize/prune khôngđồngnghĩa mọiedgehợplýhoặccoverđầyđủ. Chỉpositiveannotations, chưadownload/stageactualscenegraphs, chưabiếtretainedCOCO/ontology/splitcount.
- [VSR,TACL2023](https://arxiv.org/pdf/2205.00363): §3.1–3.4, validationpolicy vàTable2. Humanbinarylabels; **lọcmultipleinstances của queriedclasses**, nênimagelevelobservedlabelkhôngchotấtdặtpairtruth khireferentsambiguous. Originalrandomsplit7,680/1,097/2,195instances; khôngnhậnlàimage-groupedsplit hoặcsourceđãđủlocalcoverage. OfficialrepoREADMEdirectlinkmain404; khôngcoiđãđọcsourceREADME.
- [Winoground,CVPR2022](https://arxiv.org/pdf/2204.03162): §3.1schema/annotation/tags/Fig3. Human2image/2captionpreferences vàsamewordsdifferentorder cóthểlàmsecondarycompositioncheck. **Khônglàrequiredpilot hoặc đãavailable**, chưakiểmaccess/parseeligibility; khôngclaim400examplesđềutestsame-referentmulti-claim. ProposalchọnGQAadaptationheldpositivewitness cho minimalbindingcheck.

## Retrieval / cache receipt

Web search/open503; fallbackOpenAlexAPI đểdiscovermetadata, sauđófetchprimaryPDFtrựctiếp bằngurllibProxyHandler({}) vìlocalhostproxyrefuse. Khôngthaynetworkconfig, khôngmodel/extraction. Academic-searchskill dùngmulti-source-search; PDFskillchoPDFreading. KhôngtríchcơchếtừOpenAlexabstract.

Cache gốc ngoài repo: `C:/Users/leduc/OneDrive/Tài liệu/ChatGPT/Đề cương/tmp/pdfs/alignment_restart_20261030/`. MỗiPDF cótextcùngbasename. Hashes:

```text
johnson_sg2015.pdf 2ceda199b345694b3a8277bc7da219dc5a21942e94ce33bdc870ed477625cabc
mattnet.pdf 83556b0f48cf878e0cf99a0722ad69ab86e38a4ad0bdc1323b06fe75c227e790
pragmatic_clip.pdf 89f9f3a2e5bd6167c87bb347b5726bb8ffbe36150c7f44a62f12fbbce49ed2c4
comclip.pdf a31df36a35d2176557879154f198cbdbf49c438ed66ef00f09ea6ad424679dd4
cora.pdf 58931b8b095dd9028c3d2e06014ad34cfecc4cb6e1261205d211150ec7ae9a37
dino_shout.pdf 75d97052b70366526c68f25b59e40191f838c9b55a01a8299b9d3fa6a81d4544
mocha.pdf 1c6d7eed912f8b51f1b736d8c07955f66c0ad1fac849bbef27de7fb38ba231cc
lure.pdf 70757990b1bd466612d6c5311ed8410b7b1f9cbeb4f26dcd09017940acb7d503
gqa.pdf 327aae4222748426a66c145a9b396dc0b8e7eb63cead13d6d6917046a5470d49
vsr.pdf 4cd8252cee7cdec87e08a23b5c0233aab32004b31a628fc2954f5dd0bc0f949b
winoground.pdf ca74ebefb947d164b799b0b2db61baeb7e6055381e96cc163384afc0d1c82bbe
```

MộtguessedarxivID1505.00239 trả vềbàiToánkháctitle, đã**loại**, khôngđọcmethod/khôngdùnglàJohnsonprior. Filecache `sg_retrieval.pdf`/sources_structured.json giữreceipt lỗiđểkhôngnhầmvềsau. Johnsonđượcxácminh lại bằngđúngCVFPDF/title/hashởtrên. Downloadkhôngđồngnghĩafullpaperread: chỉreadscopesđãghi, chưafullappendices/experiments/rerun.

Localhistoryđãđọcđểkhônglặplạiclaim: LEARNED_CAPTION_RANKER_PILOT[_RESULT].md (CIDErpreference/global+tokenpatchattention); CAPTION_CANDIDATE_HEADROOM_RESULT.md (checkpointkhác, khôngownC0headroom); VGregionG/GV,framepretraining/sharedreadoutsourceoverview. KhôngđọclạiZIP/labels175. Khôngcommit/push/sheetreceipt trongreadingnotesnày;rootphânphối sau review.
