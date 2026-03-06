import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class PVVisualizer:
    """光伏数据可视化类"""
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
    def plot_power_time_series(self, data, save_path=None):
        """绘制功率时间序列图"""
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # 实际功率 vs 理论功率
        axes[0].plot(data.index, data['realPower'], label='实际功率', alpha=0.7, color=self.colors[0])
        axes[0].plot(data.index, data['theoretical_power'], label='理论功率', alpha=0.7, color=self.colors[1])
        axes[0].set_ylabel('功率 (MW)')
        axes[0].set_title('光伏电站发电功率时间序列')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 性能比
        axes[1].plot(data.index, data['performance_ratio'], label='性能比', alpha=0.7, color=self.colors[2])
        axes[1].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='理想性能比')
        axes[1].set_ylabel('性能比')
        axes[1].set_xlabel('时间')
        axes[1].set_title('性能比时间序列')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_seasonal_analysis(self, data, save_path=None):
        """绘制季节性分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 月度平均功率
        monthly_avg = data.groupby(data.index.month)['realPower'].mean()
        axes[0,0].bar(monthly_avg.index, monthly_avg.values, color=self.colors[0], alpha=0.7)
        axes[0,0].set_xlabel('月份')
        axes[0,0].set_ylabel('平均功率 (MW)')
        axes[0,0].set_title('月度平均发电功率')
        axes[0,0].grid(True, alpha=0.3)
        
        # 小时平均功率
        hourly_avg = data.groupby(data.index.hour)['realPower'].mean()
        axes[0,1].plot(hourly_avg.index, hourly_avg.values, marker='o', color=self.colors[1])
        axes[0,1].set_xlabel('小时')
        axes[0,1].set_ylabel('平均功率 (MW)')
        axes[0,1].set_title('日内平均发电功率')
        axes[0,1].grid(True, alpha=0.3)
        
        # 季节性热力图
        pivot_data = data.pivot_table(values='realPower', 
                                    index=data.index.hour, 
                                    columns=data.index.month, 
                                    aggfunc='mean')
        sns.heatmap(pivot_data, ax=axes[1,0], cmap='YlOrRd', cbar_kws={'label': '功率 (MW)'})
        axes[1,0].set_xlabel('月份')
        axes[1,0].set_ylabel('小时')
        axes[1,0].set_title('功率季节性热力图')
        
        # 容量因子分布
        daily_cf = data.groupby(data.index.date).apply(
            lambda x: x['realPower'].sum() * 0.25 / (100 * 24)  # 15min数据转日容量因子
        )
        axes[1,1].hist(daily_cf, bins=30, alpha=0.7, color=self.colors[2], edgecolor='black')
        axes[1,1].axvline(daily_cf.mean(), color='red', linestyle='--', 
                         label=f'平均值: {daily_cf.mean():.3f}')
        axes[1,1].set_xlabel('日容量因子')
        axes[1,1].set_ylabel('频数')
        axes[1,1].set_title('日容量因子分布')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_weather_correlation(self, data, save_path=None):
        """绘制天气因素相关性分析"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 辐照度 vs 功率
        daytime_data = data[data['is_daytime']]
        axes[0,0].scatter(daytime_data['irradiance'], daytime_data['realPower'], 
                         alpha=0.5, s=1, color=self.colors[0])
        axes[0,0].set_xlabel('辐照度 (W/m²)')
        axes[0,0].set_ylabel('实际功率 (MW)')
        axes[0,0].set_title('辐照度 vs 功率')
        axes[0,0].grid(True, alpha=0.3)
        
        # 温度 vs 性能比
        axes[0,1].scatter(daytime_data['temperature'], daytime_data['performance_ratio'], 
                         alpha=0.5, s=1, color=self.colors[1])
        axes[0,1].set_xlabel('温度 (°C)')
        axes[0,1].set_ylabel('性能比')
        axes[0,1].set_title('温度 vs 性能比')
        axes[0,1].grid(True, alpha=0.3)
        
        '''
        # 云量 vs 清晰指数
        axes[0,2].scatter(daytime_data['cloudiness'], daytime_data['clear_sky_index'], 
                         alpha=0.5, s=1, color=self.colors[2])
        axes[0,2].set_xlabel('云量 (%)')
        axes[0,2].set_ylabel('清晰指数')
        axes[0,2].set_title('云量 vs 清晰指数')
        axes[0,2].grid(True, alpha=0.3)
        '''
        # 湿度分布
        axes[1,0].hist(data['humidity'], bins=30, alpha=0.7, color=self.colors[3], edgecolor='black')
        axes[1,0].set_xlabel('湿度 (%)')
        axes[1,0].set_ylabel('频数')
        axes[1,0].set_title('湿度分布')
        axes[1,0].grid(True, alpha=0.3)
        
        # 风速分布
        axes[1,1].hist(data['windSpeed'], bins=30, alpha=0.7, color=self.colors[4], edgecolor='black')
        axes[1,1].set_xlabel('风速 (m/s)')
        axes[1,1].set_ylabel('频数')
        axes[1,1].set_title('风速分布')
        axes[1,1].grid(True, alpha=0.3)
        
        # 气压变化
        axes[0,2].plot(data.index[:1000], data['pressure'][:1000], color=self.colors[5])
        axes[0,2].set_xlabel('时间')
        axes[0,2].set_ylabel('气压 (hPa)')
        axes[0,2].set_title('气压变化（前1000个点）')
        axes[0,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_prediction_results(self, y_true, y_pred, method_name, save_path=None):
        """绘制预测结果对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 时间序列对比
        time_index = range(len(y_true))
        axes[0,0].plot(time_index, y_true, label='实际值', alpha=0.7, color=self.colors[0])
        axes[0,0].plot(time_index, y_pred, label='预测值', alpha=0.7, color=self.colors[1])
        axes[0,0].set_xlabel('时间点')
        axes[0,0].set_ylabel('功率 (MW)')
        axes[0,0].set_title(f'{method_name} - 预测vs实际')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 散点图
        axes[0,1].scatter(y_true, y_pred, alpha=0.5, s=10, color=self.colors[2])
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        axes[0,1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        axes[0,1].set_xlabel('实际功率 (MW)')
        axes[0,1].set_ylabel('预测功率 (MW)')
        axes[0,1].set_title('预测vs实际散点图')
        axes[0,1].grid(True, alpha=0.3)
        
        # 误差分布
        errors = y_pred - y_true
        axes[1,0].hist(errors, bins=30, alpha=0.7, color=self.colors[3], edgecolor='black')
        axes[1,0].axvline(errors.mean(), color='red', linestyle='--', 
                         label=f'平均误差: {errors.mean():.3f}')
        axes[1,0].set_xlabel('误差 (MW)')
        axes[1,0].set_ylabel('频数')
        axes[1,0].set_title('预测误差分布')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 误差时间序列
        axes[1,1].plot(time_index, errors, alpha=0.7, color=self.colors[4])
        axes[1,1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1,1].set_xlabel('时间点')
        axes[1,1].set_ylabel('误差 (MW)')
        axes[1,1].set_title('预测误差时间序列')
        axes[1,1].grid(True, alpha=0.3)
        
        # 计算评价指标
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = stats.pearsonr(y_true, y_pred)[0]**2
        
        fig.suptitle(f'{method_name} - MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}', 
                    fontsize=14, y=0.98)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        
    def plot_stl_decomposition(self, data, column='realPower', save_path=None):
        """绘制STL分解图"""
        from statsmodels.tsa.seasonal import STL
        
        # 使用日均数据进行STL分解
        daily_data = data.groupby(data.index.date)[column].mean()
        daily_data.index = pd.to_datetime(daily_data.index)
        
        # STL分解
        stl = STL(daily_data, seasonal=13, period=365)  # 假设年周期
        result = stl.fit()
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        # 原始序列
        axes[0].plot(result.observed, color=self.colors[0])
        axes[0].set_ylabel('原始数据')
        axes[0].set_title('STL分解 - 日平均功率')
        axes[0].grid(True, alpha=0.3)
        
        # 趋势项
        axes[1].plot(result.trend, color=self.colors[1])
        axes[1].set_ylabel('趋势项')
        axes[1].grid(True, alpha=0.3)
        
        # 季节项
        axes[2].plot(result.seasonal, color=self.colors[2])
        axes[2].set_ylabel('季节项')
        axes[2].grid(True, alpha=0.3)
        
        # 残差项
        axes[3].plot(result.resid, color=self.colors[3])
        axes[3].set_ylabel('残差项')
        axes[3].set_xlabel('时间')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_autocorrelation(self, data, column='realPower', lags=200, save_path=None):
        """绘制自相关和偏自相关图"""
        from statsmodels.tsa.stattools import acf, pacf
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 计算自相关
        autocorr = acf(data[column].dropna(), nlags=lags, fft=True)
        
        # 绘制自相关图
        axes[0].plot(range(len(autocorr)), autocorr, color=self.colors[0])
        axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[0].axhline(y=0.05, color='red', linestyle='--', alpha=0.5)
        axes[0].axhline(y=-0.05, color='red', linestyle='--', alpha=0.5)
        axes[0].set_xlabel('滞后期')
        axes[0].set_ylabel('自相关系数')
        axes[0].set_title('自相关函数 (ACF)')
        axes[0].grid(True, alpha=0.3)
        
        # 计算偏自相关
        partial_autocorr = pacf(data[column].dropna(), nlags=lags)
        
        # 绘制偏自相关图
        axes[1].plot(range(len(partial_autocorr)), partial_autocorr, color=self.colors[1])
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1].axhline(y=0.05, color='red', linestyle='--', alpha=0.5)
        axes[1].axhline(y=-0.05, color='red', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('滞后期')
        axes[1].set_ylabel('偏自相关系数')
        axes[1].set_title('偏自相关函数 (PACF)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_model_comparison(self, results_dict, save_path=None):
        """绘制多个模型的性能对比"""
        methods = list(results_dict.keys())
        metrics = ['MAE', 'RMSE', 'R2']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            values = [results_dict[method][metric] for method in methods]
            bars = axes[i].bar(methods, values, color=self.colors[:len(methods)], alpha=0.7)
            axes[i].set_ylabel(metric)
            axes[i].set_title(f'模型{metric}对比')
            axes[i].grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(values),
                           f'{value:.3f}', ha='center', va='bottom')
            
            # 旋转x轴标签
            plt.setp(axes[i].get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show() 