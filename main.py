from fastapi import FastAPI
from starlette.requests import Request
from harvesttext import HarvestText

import uvicorn

ht = HarvestText()
app = FastAPI()


@app.get('/')
async def root():
    return {1: 1}


@app.post('/api/kw-extraction')
async def get_orc(request: Request):
    """
    curl -XPOST http://localhost:8000/api/ocr \
        -H 'Content-Type: applicaton/json' \
        -d '{"text": "备受社会关注的湖南常德滴滴司机遇害案，将于1月3日9时许，在汉寿县人民法院开庭审理。此前，犯罪嫌疑人、19岁大学生杨某淇被鉴定为作案时患有抑郁症，为“有限定刑事责任能力”。\n新京报此前报道，2019年3月24日凌晨，滴滴司机陈师傅，搭载19岁大学生杨某淇到常南汽车总站附近。坐在后排的杨某淇趁陈某不备，朝陈某连捅数刀致其死亡。事发监控显示，杨某淇杀人后下车离开。随后，杨某淇到公安机关自首，并供述称“因悲观厌世，精神崩溃，无故将司机杀害”。据杨某淇就读学校的工作人员称，他家有四口人，姐姐是聋哑人。\n今日上午，田女士告诉新京报记者，明日开庭时间不变，此前已提出刑事附带民事赔偿，但通过与法院的沟通后获知，对方父母已经没有赔偿的意愿。当时按照人身死亡赔偿金计算共计80多万元，那时也想考虑对方家庭的经济状况。\n田女士说，她相信法律，对最后的结果也做好心理准备。对方一家从未道歉，此前庭前会议中，对方提出了嫌疑人杨某淇作案时患有抑郁症的辩护意见。另具警方出具的鉴定书显示，嫌疑人作案时有限定刑事责任能力。\n新京报记者从陈师傅的家属处获知，陈师傅有两个儿子，大儿子今年18岁，小儿子还不到5岁。“这对我来说是一起悲剧，对我们生活的影响，肯定是很大的”，田女士告诉新京报记者，丈夫遇害后，他们一家的主劳动力没有了，她自己带着两个孩子和两个老人一起过，“生活很艰辛”，她说，“还好有妹妹的陪伴，现在已经好些了。”"}'
    """
    data = await request.json()
    res = ht.extract_keywords(data['text'], 4)
    return {
        'ok': True,
        'data': res
    }

if __name__ == "__main__":
    uvicorn.run(app="main:app", host="0.0.0.0", port=8000, reload=True, debug=True)
