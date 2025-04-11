cmake_minimum_required(VERSION 3.10)
project(CryptoLSTMTrader)

set(CMAKE_CXX_STANDARD 17)

include_directories(
    ${CMAKE_SOURCE_DIR}/vendor/websocketpp
    ${CMAKE_SOURCE_DIR}/utils
)

find_package(Boost REQUIRED system)

add_executable(crypto_lstm_trader
    main.cpp
    data_feed/data_feed.cpp
    data_feed/fix_parser.cpp
    executor/trade_executor.cpp
    order_book/order_book.cpp
    order_book/order_matcher.cpp
    utils/logger.cpp
)

target_link_libraries(crypto_lstm_trader
    Boost::system
    ssl
    crypto
    pthread
)
