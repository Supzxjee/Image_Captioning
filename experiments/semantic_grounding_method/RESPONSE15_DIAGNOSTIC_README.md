# Chạy notebook C0 trên15cặp đã xác nhận

Notebook: [SEMANTIC_RESPONSE_C0_15_DIAGNOSTIC.ipynb](SEMANTIC_RESPONSE_C0_15_DIAGNOSTIC.ipynb).

1. Tải gói local semantic_response_confirmed15_assets.zip (3,464,452bytes), tạo một **Kaggle Dataset private** từgói này. Giữ tênZIP hoặc giữ file semantic_response_confirmed15_manifest.json và thưmụcimages saugiảinén; không cần tảiMED5GB.
2. Upload notebook vàoKaggle. AddInput3dataset: gói15cặpvừa tạo, concept-mass-c-c0-v111-v112, prompt-cache.
3. InternetON vàGPU. Notebook dùngGPU0; vớiT4×2GPU1khôngdùng vì30ảnhkhôngcầnDDP. Kiểm2path CHECKPOINT/LEGACY_PROMPT ởcellđầu. ASSETS_INPUT mặcđịnh/kaggle/input tựtìm gói; nếu gắn nhiều bản thìđiềnđúngrootdataset.
4. Save&RunAll. Khôngtrain, khôngthaycheckpoint. Sourcecommit/filehashes vàmodel/dataidentities được kiểmtrước probe.
5. Gửi semantic_response_C0_15_review.zip trongOutput. TrongZIP có summary.json, records.json, ảnh/caption HTML, source vàlog. Nếubootstrap/workerlỗi, gửiZIP/log; statusINCOMPLETE không được coi là kếtquảđủ15cặp.

Gói local ở experiments/semantic_grounding_method/review/semantic_response_confirmed15_assets.zip; ảnhkhôngpublicGitHub. [Receipt](RESPONSE15_ASSET_RECEIPT.json) ghi fullhash. Không cần thêmCOCOimages/visualcache/UPT/detector hoặcchairmetricdatasets.

## Nó đo gì

- C0 cốđịnh, cùngpromptgốc choảnhtrước/sau. Decoder rank cụmChangedbefore/after cóđổi đúngchiều vàđúngcảhaiảnh không?
- FactStable cònđược ưu tiên ởcảhaiảnh không? Độbiếnđộng contrast bao nhiêu?
- Captiongreedy thực tế viếtgì trênmỗiảnh? Khácchữ khôngđồngnghĩađúngngữnghĩa; khôngtựchấm bằng việc tìm màu trongcâu.

Đây là kiểmtracơchế banđầu trên15developmentpairs. Editedside cốýgiữpromptcũ, khôngphải fullpipeline táitạoprompt. KhôngđoCIDEr/CHAIRgain, khôngchứngminhnovelty haykhả nănggeneralization.

## Kiểmtra local đã làm

4unittests chỉmetadata/mockPASS (khôngloadTorch/transformers); góiZIPthật/manifest/30imagehashesPASS; ASTsource vànotebookcells kiểm tĩnh. ChưaCUDA/model; người dùng tựchạyKaggle. Notebookkhôngđượcagentexecute/submit.
