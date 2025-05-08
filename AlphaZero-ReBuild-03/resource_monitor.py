import torch
import psutil
import pynvml
import time
from datetime import datetime

class ResourceMonitor:
    def __init__(self, interval=1):
        self.interval = interval
        self.gpu_available = torch.cuda.is_available()
        
        if self.gpu_available:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception as e:
                print(f"GPU监控初始化失败: {str(e)}")
                self.gpu_available = False

    def _get_gpu_stats(self):
        if not self.gpu_available:
            return None, None, None, None, None
        
        try:
            # 显存信息
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            gpu_mem_used = mem_info.used / (1024**3)  # GB
            gpu_mem_total = mem_info.total / (1024**3)  # GB
            gpu_mem_percent = (mem_info.used / mem_info.total) * 100
            
            # 利用率信息
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            gpu_util = util.gpu  # GPU计算单元利用率
            
            # 温度和功耗
            gpu_temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            gpu_power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000  # 瓦特
            
            return gpu_util, gpu_mem_used, gpu_mem_total, gpu_mem_percent, gpu_temp, gpu_power
        except Exception as e:
            print(f"获取GPU信息失败: {str(e)}")
            return None, None, None, None, None, None

    def _format_line(self, label, value, indent=0):
        """格式化单行输出，保持对齐"""
        indent_str = " " * indent
        return f"| {label:<9} | {indent_str}{value:<47} |"

    def monitor(self):
        try:
            while True:
                # 清屏
                print("\033c", end="")
                
                # 获取时间戳
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                # 获取CPU信息
                cpu_total = psutil.cpu_percent()
                cpu_cores = psutil.cpu_percent(percpu=True)
                
                # 获取内存信息
                mem = psutil.virtual_memory()
                mem_used = mem.used / (1024**3)
                mem_total = mem.total / (1024**3)
                mem_percent = mem.percent
                
                # 获取GPU信息
                gpu_stats = self._get_gpu_stats()
                
                # 打印时间
                print(f"时间 {timestamp}")
                print()

                # 打印CPU信息
                print("CPU总负载    CPU核心负载")
                core_lines = []
                current_line = []
                for i, core in enumerate(cpu_cores):
                    current_line.append(f"{core:.1f}%")
                    if len(current_line) == 5:
                        core_lines.append("\t".join(current_line))
                        current_line = []
                if current_line:
                    core_lines.append(" ".join(current_line))

                print(f"  {cpu_total:.1f}%     {core_lines[0]}")
                for line in core_lines[1:]:
                    print(f"            {line}")
                print()

                # 打印内存信息
                mem_str = f"{mem_used:.2f}GB/{mem_total:.0f}GB     {mem_percent:.1f}%"
                print("  内存用量       内存%")
                print(f"{mem_str[:18]} {mem_str[19:]}")
                print()

                # 打印GPU信息
                if self.gpu_available and all(x is not None for x in gpu_stats):
                    gpu_util, gpu_mem_used, gpu_mem_total, gpu_mem_percent, gpu_temp, gpu_power = gpu_stats
                    print("GPU计算%     显存用量     显存%     温度     功耗")
                    gpu_line = (
                        f" {gpu_util:.1f}%  "
                        f"    {gpu_mem_used:.1f}GB/{gpu_mem_total:.0f}GB "
                        f"    {gpu_mem_percent:.1f}% "
                        f"    {gpu_temp}°C "
                        f"   {gpu_power:.1f}W"
                    )
                    print(gpu_line)
                
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            print("\n监控已停止")
        finally:
            if hasattr(self, 'handle'):
                pynvml.nvmlShutdown()

def start_monitoring():
    """启动资源监控"""
    try:
        monitor = ResourceMonitor()
        monitor.monitor()
    except Exception as e:
        print(f"监控异常: {str(e)}")

if __name__ == "__main__":
    start_monitoring()