def combine_score(radar: dict) -> int:
    weights = {"smoothness":0.25,"redness":0.15,"tone":0.20,"oiliness":0.15,"eyebag":0.10,"acne":0.15}
    return int(sum(radar[k]*w for k,w in weights.items()))

def skin_type_from(radar: dict) -> str:
    if radar["oiliness"] < 40: return "ผิวแห้ง"
    if radar["oiliness"] > 70 and radar["redness"] > 60: return "ผิวมัน"
    return "ผิวผสม"

def summarize(radar: dict, total: int):
    highlights, improvements = [], []
    if radar["smoothness"]>75: highlights.append("ผิวเรียบเนียน")
    if radar["oiliness"]>60:   highlights.append("สมดุลความมันดี")
    if radar["redness"]<60:    improvements.append("ผิวแดงเล็กน้อย")
    if radar["acne"]<60:       improvements.append("สิว/จุดอุดตัน")
    summary = "ผิวสุขภาพดีมาก ✨" if total>=80 else "ผิวปกติดี 😊"
    return highlights, improvements, summary
