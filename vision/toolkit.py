from decimal import Decimal


class ToolKit:
    @staticmethod
    def index_number(li, default_num):
        """从列表中选取最接近者（面向列表）"""
        select = Decimal(str(default_num)) - Decimal(str(li[0]))
        index = 0
        for i in range(1, len(li) - 1):
            select2 = Decimal(str(default_num)) - Decimal(str(li[i]))
            if abs(select) > abs(select2):
                select = select2
                index = i
        return {"a": li[index], "index": index}
