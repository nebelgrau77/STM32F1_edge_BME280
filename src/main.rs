//! simple weather station using a Bosch BME280 sensor
//! and an SSD1306 OLED display
//! 
//! in order to share the I2C bus between the two devices
//! the shared-bus crate is used
//! 
//! this project uses the STM32F103C8T6 "blue pill" board
//! 
//! the BME280 driver requires BlockingI2c trait
//! available in the the STM32F1xx-HAL crate but not in STM32F0xx nor STM32F4xx
//! 
//! as the BME280 initialization consumes the delay instance
//! the delay in the loop is done blocking the program for n instructions


#![no_std]
#![no_main]

extern crate cortex_m_rt as rt;
extern crate panic_halt;
extern crate stm32f1xx_hal as hal;
extern crate shared_bus;
extern crate cortex_m;

use bme280::BME280;

use cortex_m_rt::entry;

use hal::{
    i2c::{BlockingI2c, DutyCycle, Mode},
    prelude::*,
    stm32,
    delay::Delay,
    
};

use embedded_graphics::{
    fonts::{Font12x16, Text},
    pixelcolor::BinaryColor,
    prelude::*,
    style::TextStyleBuilder,
    };

use ssd1306::{prelude::*, Builder as SSD1306Builder};

use core::fmt;
use core::fmt::Write;
use arrayvec::ArrayString;

static UPPER: [f32; 3] = [-0.09970853,  0.0309918, -0.02518554]; //W0, W1, b; values above this line are BAD 
static LOWER: [f32; 3] = [0.06826359, -0.04702842, -0.77952247]; //W0, W1, b; values below this line are GOOD

const BOOT_DELAY_MS: u16 = 100; 

#[entry]
fn main() -> ! {
    let dp = stm32::Peripherals::take().unwrap();
    let cp = cortex_m::Peripherals::take().unwrap();

    let mut flash = dp.FLASH.constrain();
    let mut rcc = dp.RCC.constrain();

    let clocks = rcc.cfgr.use_hse(8.mhz()).sysclk(8.mhz()).freeze(&mut flash.acr);

    // delay provider for the BME280
    let mut delay = Delay::new(cp.SYST, clocks);
    
    delay.delay_ms(BOOT_DELAY_MS);



    let mut afio = dp.AFIO.constrain(&mut rcc.apb2);

    let mut gpiob = dp.GPIOB.split(&mut rcc.apb2);

    let scl = gpiob.pb8.into_alternate_open_drain(&mut gpiob.crh);
    let sda = gpiob.pb9.into_alternate_open_drain(&mut gpiob.crh);
    
    
    let i2c = BlockingI2c::i2c1(
        dp.I2C1,
        (scl, sda),
        &mut afio.mapr,
        Mode::Fast {
            frequency: 400_000.hz(),
            duty_cycle: DutyCycle::Ratio2to1,
        },
        clocks,
        &mut rcc.apb1,
        1000,
        10,
        1000,
        1000,
    );

    // shared-bus manager created
    let manager = shared_bus::CortexMBusManager::new(i2c);
    
    
    // BME280 sensor initiation
    let bme280_i2c_addr = 0x76;
    let mut bme280 = BME280::new(manager.acquire(), bme280_i2c_addr, delay);    
    bme280.init().unwrap();
    
    //ssd1306 i2c address: not required in this case
    //let ssd1306_i2c_addr = 0x3c;

    // display initiated in TerminalMode
    let mut disp: GraphicsMode<_> = SSD1306Builder::new().connect_i2c(manager.acquire()).into();
        
    disp.init().unwrap();
    
    
    let text_style = TextStyleBuilder::new(Font12x16).text_color(BinaryColor::On).build();


    loop {
        
        for x in 0..128 {
            for y in 0..56 {
                disp.set_pixel(x,y,0);
            }
        }


        let text_style = TextStyleBuilder::new(Font12x16).text_color(BinaryColor::On).build();
        let mut buf_temp = ArrayString::<[u8; 9]>::new();
        let mut buf_hum = ArrayString::<[u8; 9]>::new();
        let mut text_status = ArrayString::<[u8; 16]>::new();
        
        //get values from the sensor
        let measurements = bme280.measure().unwrap();

        let temperature = measurements.temperature;
        let humidity = measurements.humidity;

        let h_sucks = -(temperature * UPPER[0] + UPPER[2]) / UPPER[1];
        let h_nice = -(temperature * LOWER[0] + LOWER[2]) / LOWER[1];


        format(&mut buf_temp, (temperature * 10.0) as u16, 84 as char, 67 as char);

        Text::new(buf_temp.as_str(), Point::new(0, 0)).into_styled(text_style).draw(&mut disp);

        format(&mut buf_hum, (humidity * 10.0) as u16, 72 as char, 37 as char);
    
        Text::new(buf_hum.as_str(), Point::new(0, 16)).into_styled(text_style).draw(&mut disp);

        let mut status: u8 = 1;
        
        if humidity >= h_sucks {
            status = 0 }
        else if humidity <= h_nice {
            status = 2 }
        else {
            status = 1
        }
        
        status_display(&mut text_status, status);
        Text::new(text_status.as_str(), Point::new(0, 40)).into_styled(text_style).draw(&mut disp);

        disp.flush().unwrap();

        //delay.delay_ms(50_u16);
        //should be 1 ms...
        cortex_m::asm::delay(1 * 8_000_000);


    }

}

// helper function to display temperature and humidity

fn format(buf: &mut ArrayString<[u8; 9]>, val: u16, feature: char, unit: char) {
    
    let tenths = val%10;
    let singles = (val/10)%10;
    let tens = (val/100)%10;
        
    fmt::write(buf, format_args!("{}: {}{}.{} {}", 
    feature, tens as u8, singles as u8, tenths as u8, unit)).unwrap();
}

// helper function to display the weather status

fn status_display(buf: &mut ArrayString<[u8;16]>, status: u8) {
    if status == 2 {
        fmt::write(buf, format_args!("IT'S NICE! :)   ")).unwrap();
    }
    else if status == 0 {
        fmt::write(buf, format_args!("BUMMER :(       ")).unwrap();
    } 
    else {
        fmt::write(buf, format_args!("IT'S OK.        ")).unwrap();
    }
}