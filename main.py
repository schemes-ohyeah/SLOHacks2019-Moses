import json

import numpy


def moses(request):
    if request.method == "OPTIONS":
        return do_options()
    elif request.method == "POST":
        return do_post(request)
    else:
        return "Bad method", 405


def do_options():
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "*",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Max-Age": "3600"
    }
    return "", 204, headers


def do_post(request):
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Content-Type": "application/json"
    }
    request_json = request.get_json()
    reference_data = request_json.get("reference")
    recent_data = request_json.get("recent")

    fitx, fity, fitz = fit_to_time_functions(reference_data)

    recent_x = extract_vector(recent_data, 0)
    recent_y = extract_vector(recent_data, 1)
    recent_z = extract_vector(recent_data, 2)

    error_x = find_error(fitx, recent_x)
    error_y = find_error(fity, recent_y)
    error_z = find_error(fitz, recent_z)

    return json.dumps({
        "error_x": error_x,
        "error_y": error_y,
        "error_z": error_z
    }), 201, headers


def fit_to_time_functions(tuple_list):
    """
    Fits a data set to a x, y, and z polynomial functions of time.
    We know that the samples are equally spaced in time
    so we can arbitrarily use numpy.arange() to mock time intervals

    :param tuple_list: A list of size 3 tuples representing (x, y, z)
    :return: fitx, fity, fitz each of them as a axis as a function of time
    """
    x_vector = extract_vector(tuple_list, 0)
    y_vector = extract_vector(tuple_list, 1)
    z_vector = extract_vector(tuple_list, 2)
    t = numpy.arange(len(x_vector))

    degrees = 4
    fitx = numpy.polyfit(t, x_vector, degrees)
    fity = numpy.polyfit(t, y_vector, degrees)
    fitz = numpy.polyfit(t, z_vector, degrees)

    return fitx.tolist()[:-1], fity.tolist()[:-1], fitz.tolist()[:-1]


def extract_vector(tuple_list, index):
    """
    Takes a index from the list of 3 tuples and flattens it

    :param tuple_list: list of 3 tuples
    :param index: 0 | 1 | 2 representing x, y, z
    :return: List[float]
    """
    return [x[index] for x in tuple_list]


def find_error(c, values):
    """
    1. Calculates the ranges between the fit function and the actual measurements.
    2. Returns standard deviation of the list of ranges

    :param c: list of coefficients
    :param values: recorded measurements
    :return: standard deviation of range difference
    """
    def position_function(coef, t):
        return coef[0] * (t ** 4) + coef[1] * (t ** 3) + coef[2] * (t ** 2) + coef[3] * t

    diffs = []
    for t in range(len(values)):
        diffs.append(position_function(c, t) - values[t])

    return numpy.std(diffs)
