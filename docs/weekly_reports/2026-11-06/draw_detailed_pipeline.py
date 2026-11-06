"""Draw the audited implemented pipeline; no model execution."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from html import escape
import math
P=Path(__file__).resolve().parent
W,H=3000,2480
im=Image.new('RGB',(W,H),'white');d=ImageDraw.Draw(im);svg=[]
C='#173D5B';ink='#172B3A';muted='#536574'
fonts={n:ImageFont.truetype('C:/Windows/Fonts/arial.ttf',n) for n in [25,28,30,32,36,40,48,58]}
def rect(x,y,w,h,fill='#EFF5FA',stroke=C,dash=False):
 d.rounded_rectangle((x,y,x+w,y+h),radius=18,fill=fill,outline=stroke,width=3)
 svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="{fill}" stroke="{stroke}" stroke-width="3"'+(' stroke-dasharray="12 8"' if dash else '')+'/>')
def txt(x,y,s,size=32,color=ink,bold=False):
 f=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',size) if bold else fonts.setdefault(size,ImageFont.truetype('C:/Windows/Fonts/arial.ttf',size))
 for j,line in enumerate(s.split('\n')):
  yy=y+j*(size+7);d.text((x,yy),line,font=f,fill=color)
  svg.append(f'<text x="{x}" y="{yy+size}" font-family="Arial" font-size="{size}" fill="{color}"'+(' font-weight="bold"' if bold else '')+'>'+escape(line)+'</text>')
def arrow(pts,color=C,dashed=False):
 d.line(pts,fill=color,width=4)
 x,y=pts[-1];a,b=pts[-2];t=math.atan2(y-b,x-a)
 poly=[(x,y),(x-17*math.cos(t-.45),y-17*math.sin(t-.45)),(x-17*math.cos(t+.45),y-17*math.sin(t+.45))]
 d.polygon(poly,fill=color)
 svg.append('<polyline points="'+' '.join(f'{a},{b}' for a,b in pts)+f'" fill="none" stroke="{color}" stroke-width="4"'+(' stroke-dasharray="12 8"' if dashed else '')+'/>')
 svg.append('<polygon points="'+' '.join(f'{a:.2f},{b:.2f}' for a,b in poly)+f'" fill="{color}"/>')
def box(x,y,w,h,title,body,fill='#EFF5FA',size=32):
 rect(x,y,w,h,fill)
 title_size=36
 while d.textlength(title,font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',title_size))>w-48:title_size-=1
 while max(d.textlength(line,font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',size)) for line in body.split('\n'))>w-48 or 68+(len(body.split('\n'))-1)*(size+7)+size>h-10:size-=1
 assert size>=24, (title,size)
 txt(x+24,y+18,title,title_size,C,True);txt(x+24,y+68,body,size)

txt(60,28,'PIPELINE CHI TIẾT: CSA + PROMPT ĐỐI TƯỢNG–QUAN HỆ',58,C,True)
txt(60,105,'Cấu hình đã chạy: XE từ đầu → SCST + CE → beam 5 → reranking cố định',36,muted)
box(60,190,900,155,'1. Ảnh đầu vào I','Cùng một ảnh được cấp cho nhánh thị giác và UPT.\nTEST: chỉ dùng ảnh; không dùng caption tham chiếu.',size=30)
box(1050,190,900,155,'2. UPT: người – hành động – vật','Box S/O, nhãn đối tượng, hành động và điểm cặp.\nUPT là nguồn dự đoán; không phải nhãn đúng tuyệt đối.',size=30)
box(2040,190,900,155,'3. Bank ngữ nghĩa cố định','Danh từ / động từ từ caption TRAIN → lemma\n→ CLIP text encoder → bank embedding 512 chiều.',size=30)
arrow([(960,265),(1050,265)])
box(60,425,900,155,'CLIP visual + box YOLO → ROI pooling','Bỏ CLS; lưới patch 14 × 14 × 768.\nPool theo box → R: N × 768; Linear → V: N × 512.',size=30)
arrow([(510,345),(510,425)])
box(1050,425,900,155,'Prompt riêng ảnh + liên kết UPT–YOLO','Lọc cặp → cụm đối tượng / quan hệ → CLIP text.\nGhép box cùng nhãn, IoU ≥ 0,5; lưu ROI / cặp (S,O).',size=30)
arrow([(1500,345),(1500,425)])
arrow([(2490,345),(2490,615),(995,615),(995,650),(930,650),(930,675)])
txt(2070,465,'Bank vẫn là một nhánh riêng,',30,muted)
txt(2070,510,'không thay cho prompt của từng ảnh.',30,muted)

rect(60,675,900,680,'#EAF3FC');txt(84,695,'4. CĂN CHỈNH CSA',40,C,True)
txt(84,758,'Semantic aggregation: mỗi bank → 150 prototype.\nBank được giữ cố định; aggregation được học.',30)
box(90,870,840,130,'Object fusion 1','V + prototype danh từ → V1, danh từ đã chọn',size=30)
box(90,1045,840,130,'Object fusion 2','V1 + danh từ đã chọn → memory đối tượng Vo',size=30)
box(90,1220,840,110,'Relation fusion','Vo + prototype động từ → memory quan hệ Vr',size=30)
arrow([(510,580),(510,675)])
arrow([(510,1000),(510,1045)]);arrow([(510,1175),(510,1220)])

rect(1050,675,900,680,'#ECF8F1');txt(1074,695,'5. HAI ADAPTER PROMPT–ROI',40,C,True)
box(1080,770,840,210,'Adapter đối tượng','Slot = MLP [text đối tượng; Vo[i]; box i]\nQ = Vo; K = V = các slot đối tượng\nMask: chỉ ROI i được đọc slot gắn với nó.\nVo′ = Vo + Wout · Attention(Q,K,V)',fill='#F8FCF9',size=30)
box(1080,1030,840,255,'Adapter quan hệ','Slot = MLP [text quan hệ; Vo[s]; Vo[o]; box S/O]\nQ = Vr; K = V = các slot quan hệ\nMask: chỉ hai ROI đầu mút được đọc slot cặp.\nVr′ = Vr + Wout · Attention(Q,K,V)\nS/O có thứ tự; mỗi slot giữ hình học của cặp.',fill='#F8FCF9',size=30)
arrow([(1500,580),(1500,675)])
arrow([(960,1000),(1020,1000),(1020,895),(1080,895)])
arrow([(1020,1000),(1020,1160),(1080,1160)])
txt(1074,1303,'Không có slot hợp lệ → residual đúng bằng 0.',28,muted)

rect(2040,675,900,425,'#F2EFFA');txt(2064,695,'6. TRANSFORMER DECODER ×3 LAYER',36,C,True)
txt(2064,760,'Token trước → embedding + positional encoding\n→ masked self-attention (chỉ nhìn từ trước)\n→ cross-attention đối tượng: Q=h; K=V=Vo′\n→ cross-attention quan hệ: Q=h; K=V=Vr′\n→ hai gate dựa trên context và embedding từ\n→ go * co + gr * cr → FFN\n→ Linear vocabulary → phân phối từ tiếp theo',30)
arrow([(1950,1000),(2040,1000)])
box(2040,1160,900,135,'Beam search = 5','Tối đa 20 token; giữ raw log-prob của 5 ứng viên.\nTự hồi quy từ BOS đến EOS.',size=30)
arrow([(2490,1100),(2490,1160)])
box(2040,1350,900,235,'7. Reranking cố định','Mỗi ứng viên có 3 điểm:\nD: raw decoder log-prob, min–max trong 5 câu\nC: CLIPScore ảnh–caption, min–max trong 5 câu\nO: nhất quán đối tượng caption với YOLO, chuẩn hóa\nS = 0,3 D + 0,4 C + 0,3 O',fill='#FFF5E8',size=30)
arrow([(2490,1295),(2490,1350)])
box(2040,1655,900,110,'CAPTION CUỐI CÙNG','argmax S; hòa điểm → ưu tiên thứ hạng beam trước.',fill='#E2F2EB',size=28)
arrow([(2490,1585),(2490,1655)])

box(60,1430,900,335,'Bên trong mỗi Fusion của CSA','1) Cosine prototype–ROI / nhiệt độ 0,1\n   → softmax theo ROI → độ liên quan prototype\n2) Self-attention thị giác + residual/norm + FFN\n3) Cross-attention: Q=thị giác; K=V=prototype chọn\n   + residual/norm + FFN\n4) Gate sigmoid trộn nhánh thị giác và ngữ nghĩa.\nĐây là bước căn chỉnh bank từ với vùng ảnh.',fill='#F7FAFD',size=30)
box(1050,1430,900,335,'Phân biệt hai lớp căn chỉnh','CSA: vùng ảnh ↔ ngữ nghĩa danh từ / động từ.\nAdapter: prompt riêng ảnh ↔ ROI / cặp ROI.\n\nKhông ghép được box → bỏ slot đó;\nkhông tự gán prompt vào toàn ảnh.\nVo′ và Vr′ vẫn là hai memory riêng cho decoder.\nReranking dùng lại ảnh và nhãn detector để chấm.',fill='#F7FCF9',size=30)

rect(60,1840,2880,445,'#FFF9ED',stroke='#9A6A18',dash=True)
txt(84,1860,'CHỈ KHI HUẤN LUYỆN — CÙNG ENCODER / ADAPTER / DECODER Ở TRÊN',40,'#785114',True)
box(90,1945,880,285,'Giai đoạn 1: XE / Cross-Entropy','Caption TRAIN dịch phải → teacher forcing\nLCE = − mean log p(từ đúng | ảnh, tiền tố)\n\nChạy 10 epoch từ khởi tạo mới của mạng học.\nCLIP và các nguồn tiền huấn luyện giữ cố định.\nChọn checkpoint theo CIDEr trên VALID. ',fill='#FFFCF5',size=30)
box(1060,1945,880,285,'Giai đoạn 2: SCST + CE','Cùng ảnh → câu lấy mẫu ys và câu greedy yg\nA = CIDEr(ys, refs TRAIN) − CIDEr(yg, refs TRAIN)\nLSCST = − mean [A · tổng log p(ys)]\nL = LCE + 0,05 LSCST\n14.161 update / toàn bộ 113.287 ảnh TRAIN\nAdam lr=0,000005; global batch=8.',fill='#FFFCF5',size=30)
box(2030,1945,880,285,'Cách dùng khi suy luận / TEST','Không cần caption tham chiếu để sinh / chọn câu.\nKhông chạy CE hoặc SCST khi suy luận.\nKhông có thưởng đối tượng trong loss lượt này.\nThành phần đối tượng O nằm ở reranking.\nCHAIR là bộ đo hậu kiểm, không phải bộ chọn.\nRefs TEST chỉ dùng tính các chỉ số sau cùng.',fill='#FFFCF5',size=30)

txt(60,2330,'Ký hiệu: N = số ROI hợp lệ; Vo/Vr = memory đối tượng/quan hệ; dấu ′ = sau adapter. Các mạng học dùng d=512, 8 head.',28,muted)
txt(60,2375,'Sơ đồ theo mã đã chạy: model.py • prompt_adapter_model.py • prompt_scratch_model.py • prompt_scst_rerank.py (06/11/2026).',28,muted)
txt(60,2420,'Đây là pipeline triển khai hiện tại, không phải tuyên bố tái hiện nguyên bản CSA-Net hoặc bảo đảm quan hệ dự đoán đều đúng.',28,muted)
im.save(P/'pipeline_chi_tiet.png')
(P/'pipeline_chi_tiet.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><rect width="100%" height="100%" fill="white"/>'+''.join(svg)+'</svg>',encoding='utf-8')
print('PNG + editable SVG written')
