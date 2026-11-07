# Visual region extraction — 07/11/2026

Người dùng chốt quy ước: modulebo góc; đầu ra vuông/tokenstack. SVGchỉnhsửađược vàPNG1800×2880, mãdựng `build.py`. Luồng dưới lên: ảnhgốc khôngbbox → CLIPViT-B/16/YOLO → patchfeatures/boundingboxcoordinates → ROIpool → Linear768→512 → V(N×512). CLIPmàuxanhriêng, Frozen; tênYOLOgiữgeneric vìproducer/checkpointversioncachechưa xácminh. Frozenởđây nghĩa không cập nhật khi huấn luyệncaption, không nói đã truynguồnđượcdetector.

Ảnh minh họa dùng asset COCO_train2014_000000263974.jpg đãcótrongrepo. Khôngvẽbbox giả lên ảnh hoặc gọi ảnh inputlàdetectoroutput. B biểu diễn tọa độđểROIpoolnhận, khôngphải ảnhđãđánhkhung. Không chạyCLIP,YOLO,Torch/model/scorer. Không sửaWord/Sheet hoặc mô hình.

QA: parseSVG,4modulebo16px, datarectrx0; renderheadlessEdge offline vàxembảnPNGcuối: chữkhôngtràn, mũitên rõhướng, ảnhI sạch. ĐãcopySVG/PNG vàoDownloads. Không cóbbox dựđoánthực tế mới được tạo trong lượt vẽ.
