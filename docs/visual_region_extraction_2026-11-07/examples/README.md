# Ảnh gốc và bbox thực từ cache — 07/11/2026

ẢnhCOCOtrain263974, assetảnhgốc cósẵn trongdocs/csa_pipeline_2026-11-02/v3/assets. Cache được tìm lại tạiC:/KhoaLuanTotNghiep/Code/.tmp/hoi_caption_validation_audit/inputs/objectdetectioncache.json. Đã băm toànfileSHA498825daf8f8773a1c92f4111dc4ccf1f4296343f952b486e0aa56caf82ec04c, khớp audit_assets.DETECTION_SHA của pipeline. Khôngchạy lại detector/model, khôngchỉnh hoặc phỏngđoán tọađộ bbox. ModelversionYOLO vẫn chưa biết.

- `image_bbox_all.svg/png`: toàn20bbox đãlưu choảnh này, màu theo nhãn, khôngghi nhãn chữ đểdễchèn.
- `image_bbox_clean.svg/png`: hiểnthị5bboxforeground đãlưu, chọnperson/bicycle conf≥.85,x2<450,area>5000. ChỉlọcHIỂNTHỊminhhọa, khôngthaycache/ROI/metric. Chúthíchhìnhphảinóiđãchọnmộtsốbbox; khônggọiđây làtoànbộdetectoroutput.
- `bbox_provenance.json`: đủ20recordnguồn,5recordhiểnthị, SHAảnh/cache.

Đãxemảnhgốckhôngbbox vàcả2bảnrender; gốc640×480, SVGcoordinates cùngảnh, PNG1280×960. Hìnhcókhung làvisualizationtọađộthựctừcache, khôngphảidựđoánmớihoặcboxvẽtay. Đãcopy3ảnhvàoDownloads. KhôngthêmSheetrow/thínghiệm hoặc đổiWord.
