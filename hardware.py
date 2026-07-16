# hardware.py
from maix import gpio, pinmap, err

pin_name = "A22" 
gpio_name = "GPIOA22"

# 確保整個系統啟動時只配置一次引腳映射
err.check_raise(pinmap.set_pin_function(pin_name, gpio_name), "set pin failed")

# 實例化唯一的 GPIO 控制對象，預設低電平（關閉）
shared_gpio = gpio.GPIO(gpio_name, gpio.Mode.OUT)
shared_gpio.value(0)