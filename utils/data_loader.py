import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

class PVDataLoader:
    """光伏数据加载和预处理类"""
    
    def __init__(self, data_path='updated_data.csv'):
        # 尝试不同的路径
        possible_paths = [
            data_path,  # 原始路径
            os.path.join('..', data_path),  # 上级目录
            os.path.join('..', '..', data_path),  # 上两级目录
            os.path.abspath(data_path),  # 绝对路径
        ]
        
        self.data_path = None
        for path in possible_paths:
            if os.path.exists(path):
                self.data_path = path
                break
        
        if self.data_path is None:
            # 如果都找不到，使用原始路径（会在后续报错）
            self.data_path = data_path
            
        self.data = None
        self.processed_data = None
        
    def load_data(self):
        """加载原始数据"""
        print("正在加载光伏数据...")
        self.data = pd.read_csv(self.data_path, encoding='utf-8')
        
        # 转换时间列
        self.data['datetime'] = pd.to_datetime(self.data['date'])
        self.data.set_index('datetime', inplace=True)
        
        # 添加时间特征
        self.data['hour'] = self.data.index.hour
        self.data['minute'] = self.data.index.minute
        self.data['day_of_year'] = self.data.index.dayofyear
        self.data['month'] = self.data.index.month
        self.data['day'] = self.data.index.day
        
        print(f"数据加载完成，共 {len(self.data)} 条记录")
        return self.data
    
    def process_data(self):
        """数据预处理"""
        if self.data is None:
            self.load_data()
            
        # 创建处理后的数据副本
        self.processed_data = self.data.copy()
        
        # 计算理论功率（基于辐照度）
        self.processed_data['theoretical_power'] = self._calculate_theoretical_power()
        
        # 计算性能比
        self.processed_data['performance_ratio'] = self._calculate_performance_ratio()
        
        # 计算清晰指数
        self.processed_data['clear_sky_index'] = self._calculate_clear_sky_index()
        
        # 添加白昼标识
        self.processed_data['is_daytime'] = self.processed_data['irradiance'] > 0
        
        # 添加时间周期特征
        self.processed_data['hour_sin'] = np.sin(2 * np.pi * self.processed_data['hour'] / 24)
        self.processed_data['hour_cos'] = np.cos(2 * np.pi * self.processed_data['hour'] / 24)
        self.processed_data['day_sin'] = np.sin(2 * np.pi * self.processed_data['day_of_year'] / 365)
        self.processed_data['day_cos'] = np.cos(2 * np.pi * self.processed_data['day_of_year'] / 365)
        
        print("数据预处理完成")
        return self.processed_data
    
    def _calculate_theoretical_power(self):
        """计算理论可发功率"""
        # 基于辐照度计算理论功率
        # 假设光伏组件效率为20%，组件面积为500平方米
        eta_ref = 270/1650/0.99  # 参考效率
        area_total = 1.6635*74000  # 总面积 (m²)
        p_rated = 20  # 额定功率 (MW，从数据中看到theoryPower都是100)
        
        # 温度修正
        gamma = -0.004  # 温度系数 (1/°C)
        t_ref = 25  # 参考温度 (°C)
        
        # 计算理论功率
        temp_factor = 1 + gamma * (self.data['temperature'] - t_ref)
        theoretical_power = (self.data['irradiance'] * eta_ref * area_total * temp_factor) / 1000  # 转换为MW
        
        # 限制在0-100MW范围内
        theoretical_power = np.clip(theoretical_power, 0, p_rated)
        
        return theoretical_power
    
    def _calculate_performance_ratio(self):
        """计算性能比"""
        theoretical = self._calculate_theoretical_power()
        real = self.data['realPower']
        
        # 避免除零错误
        performance_ratio = np.where(theoretical > 0.01, real / theoretical, 0)
        
        # 限制在合理范围内
        performance_ratio = np.clip(performance_ratio, 0, 1.2)
        
        return performance_ratio
    
    def _calculate_clear_sky_index(self):
        """计算清晰指数"""
        # 使用简化的晴空辐照模型
        # 晴空辐照 = 最大辐照 * sin(太阳高度角)
        
        # 简化计算：使用日内最大辐照作为晴空基准
        daily_max = self.data.groupby(self.data.index.date)['irradiance'].transform('max')
        
        # 清晰指数
        clear_sky_index = np.where(daily_max > 0, self.data['irradiance'] / daily_max, 0)
        clear_sky_index = np.clip(clear_sky_index, 0, 1)
        
        return clear_sky_index
    
    def split_train_test(self):
        """按要求划分训练集和测试集"""
        if self.processed_data is None:
            self.process_data()
            
        # 获取每年2、5、8、11月的最后一周作为测试集
        test_mask = pd.Series(False, index=self.processed_data.index)
        
        for year in self.processed_data.index.year.unique():
            for month in [2, 5, 8, 11]:
                # 找到每月最后一周
                month_data = self.processed_data[
                    (self.processed_data.index.year == year) & 
                    (self.processed_data.index.month == month)
                ]
                
                if len(month_data) > 0:
                    # 最后7天
                    last_week = month_data.index[-7*96:]  # 7天 * 96个15分钟间隔
                    test_mask.loc[last_week] = True
        
        train_data = self.processed_data[~test_mask]
        test_data = self.processed_data[test_mask]
        
        print(f"训练集: {len(train_data)} 条记录")
        print(f"测试集: {len(test_data)} 条记录")
        
        return train_data, test_data
    
    def get_station_info(self):
        """获取电站基本信息"""
        return {
            'rated_power': 20,  # MW
            'panel_efficiency': 270/1650/0.99,
            'panel_area': 1.6635*74000,  # m²
            'temperature_coefficient': -0.004,  # 1/°C
            'data_start': self.data.index.min(),
            'data_end': self.data.index.max(),
            'total_records': len(self.data),
            'data_resolution': '15min'
        } 