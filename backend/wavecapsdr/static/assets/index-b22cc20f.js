var __defProp = Object.defineProperty;
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __publicField = (obj, key, value) => {
  __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
  return value;
};
var __accessCheck = (obj, member, msg) => {
  if (!member.has(obj))
    throw TypeError("Cannot " + msg);
};
var __privateGet = (obj, member, getter) => {
  __accessCheck(obj, member, "read from private field");
  return getter ? getter.call(obj) : member.get(obj);
};
var __privateAdd = (obj, member, value) => {
  if (member.has(obj))
    throw TypeError("Cannot add the same private member more than once");
  member instanceof WeakSet ? member.add(obj) : member.set(obj, value);
};
var __privateSet = (obj, member, value, setter) => {
  __accessCheck(obj, member, "write to private field");
  setter ? setter.call(obj, value) : member.set(obj, value);
  return value;
};
var __privateWrapper = (obj, member, setter, getter) => ({
  set _(value) {
    __privateSet(obj, member, value, setter);
  },
  get _() {
    return __privateGet(obj, member, getter);
  }
});
var __privateMethod = (obj, member, method) => {
  __accessCheck(obj, member, "access private method");
  return method;
};
var _provider, _providerCalled, _a, _focused, _cleanup, _setup, _b, _online, _cleanup2, _setup2, _c, _gcTimeout, _d, _initialState, _revertState, _cache, _client, _retryer, _defaultOptions, _abortSignalConsumed, _dispatch, dispatch_fn, _e, _client2, _currentQuery, _currentQueryInitialState, _currentResult, _currentResultState, _currentResultOptions, _currentThenable, _selectError, _selectFn, _selectResult, _lastQueryWithDefinedData, _staleTimeoutId, _refetchIntervalId, _currentRefetchInterval, _trackedProps, _executeFetch, executeFetch_fn, _updateStaleTimeout, updateStaleTimeout_fn, _computeRefetchInterval, computeRefetchInterval_fn, _updateRefetchInterval, updateRefetchInterval_fn, _updateTimers, updateTimers_fn, _clearStaleTimeout, clearStaleTimeout_fn, _clearRefetchInterval, clearRefetchInterval_fn, _updateQuery, updateQuery_fn, _notify, notify_fn, _f, _client3, _observers, _mutationCache, _retryer2, _dispatch2, dispatch_fn2, _g, _mutations, _scopes, _mutationId, _h, _client4, _currentResult2, _currentMutation, _mutateOptions, _updateResult, updateResult_fn, _notify2, notify_fn2, _i, _queries, _j, _queryCache, _mutationCache2, _defaultOptions2, _queryDefaults, _mutationDefaults, _mountCount, _unsubscribeFocus, _unsubscribeOnline, _k;
(function polyfill() {
  const relList = document.createElement("link").relList;
  if (relList && relList.supports && relList.supports("modulepreload")) {
    return;
  }
  for (const link of document.querySelectorAll('link[rel="modulepreload"]')) {
    processPreload(link);
  }
  new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      if (mutation.type !== "childList") {
        continue;
      }
      for (const node of mutation.addedNodes) {
        if (node.tagName === "LINK" && node.rel === "modulepreload")
          processPreload(node);
      }
    }
  }).observe(document, { childList: true, subtree: true });
  function getFetchOpts(link) {
    const fetchOpts = {};
    if (link.integrity)
      fetchOpts.integrity = link.integrity;
    if (link.referrerPolicy)
      fetchOpts.referrerPolicy = link.referrerPolicy;
    if (link.crossOrigin === "use-credentials")
      fetchOpts.credentials = "include";
    else if (link.crossOrigin === "anonymous")
      fetchOpts.credentials = "omit";
    else
      fetchOpts.credentials = "same-origin";
    return fetchOpts;
  }
  function processPreload(link) {
    if (link.ep)
      return;
    link.ep = true;
    const fetchOpts = getFetchOpts(link);
    fetch(link.href, fetchOpts);
  }
})();
function getDefaultExportFromCjs(x2) {
  return x2 && x2.__esModule && Object.prototype.hasOwnProperty.call(x2, "default") ? x2["default"] : x2;
}
var jsxRuntime = { exports: {} };
var reactJsxRuntime_production_min = {};
var react = { exports: {} };
var react_production_min = {};
/**
 * @license React
 * react.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var l$1 = Symbol.for("react.element"), n$1 = Symbol.for("react.portal"), p$2 = Symbol.for("react.fragment"), q$1 = Symbol.for("react.strict_mode"), r$1 = Symbol.for("react.profiler"), t = Symbol.for("react.provider"), u = Symbol.for("react.context"), v$1 = Symbol.for("react.forward_ref"), w = Symbol.for("react.suspense"), x = Symbol.for("react.memo"), y = Symbol.for("react.lazy"), z$1 = Symbol.iterator;
function A$1(a) {
  if (null === a || "object" !== typeof a)
    return null;
  a = z$1 && a[z$1] || a["@@iterator"];
  return "function" === typeof a ? a : null;
}
var B$1 = { isMounted: function() {
  return false;
}, enqueueForceUpdate: function() {
}, enqueueReplaceState: function() {
}, enqueueSetState: function() {
} }, C$1 = Object.assign, D$1 = {};
function E$1(a, b, e) {
  this.props = a;
  this.context = b;
  this.refs = D$1;
  this.updater = e || B$1;
}
E$1.prototype.isReactComponent = {};
E$1.prototype.setState = function(a, b) {
  if ("object" !== typeof a && "function" !== typeof a && null != a)
    throw Error("setState(...): takes an object of state variables to update or a function which returns an object of state variables.");
  this.updater.enqueueSetState(this, a, b, "setState");
};
E$1.prototype.forceUpdate = function(a) {
  this.updater.enqueueForceUpdate(this, a, "forceUpdate");
};
function F() {
}
F.prototype = E$1.prototype;
function G$1(a, b, e) {
  this.props = a;
  this.context = b;
  this.refs = D$1;
  this.updater = e || B$1;
}
var H$1 = G$1.prototype = new F();
H$1.constructor = G$1;
C$1(H$1, E$1.prototype);
H$1.isPureReactComponent = true;
var I$1 = Array.isArray, J = Object.prototype.hasOwnProperty, K$1 = { current: null }, L$1 = { key: true, ref: true, __self: true, __source: true };
function M$1(a, b, e) {
  var d, c = {}, k2 = null, h = null;
  if (null != b)
    for (d in void 0 !== b.ref && (h = b.ref), void 0 !== b.key && (k2 = "" + b.key), b)
      J.call(b, d) && !L$1.hasOwnProperty(d) && (c[d] = b[d]);
  var g = arguments.length - 2;
  if (1 === g)
    c.children = e;
  else if (1 < g) {
    for (var f2 = Array(g), m2 = 0; m2 < g; m2++)
      f2[m2] = arguments[m2 + 2];
    c.children = f2;
  }
  if (a && a.defaultProps)
    for (d in g = a.defaultProps, g)
      void 0 === c[d] && (c[d] = g[d]);
  return { $$typeof: l$1, type: a, key: k2, ref: h, props: c, _owner: K$1.current };
}
function N$1(a, b) {
  return { $$typeof: l$1, type: a.type, key: b, ref: a.ref, props: a.props, _owner: a._owner };
}
function O$1(a) {
  return "object" === typeof a && null !== a && a.$$typeof === l$1;
}
function escape(a) {
  var b = { "=": "=0", ":": "=2" };
  return "$" + a.replace(/[=:]/g, function(a2) {
    return b[a2];
  });
}
var P$1 = /\/+/g;
function Q$1(a, b) {
  return "object" === typeof a && null !== a && null != a.key ? escape("" + a.key) : b.toString(36);
}
function R$1(a, b, e, d, c) {
  var k2 = typeof a;
  if ("undefined" === k2 || "boolean" === k2)
    a = null;
  var h = false;
  if (null === a)
    h = true;
  else
    switch (k2) {
      case "string":
      case "number":
        h = true;
        break;
      case "object":
        switch (a.$$typeof) {
          case l$1:
          case n$1:
            h = true;
        }
    }
  if (h)
    return h = a, c = c(h), a = "" === d ? "." + Q$1(h, 0) : d, I$1(c) ? (e = "", null != a && (e = a.replace(P$1, "$&/") + "/"), R$1(c, b, e, "", function(a2) {
      return a2;
    })) : null != c && (O$1(c) && (c = N$1(c, e + (!c.key || h && h.key === c.key ? "" : ("" + c.key).replace(P$1, "$&/") + "/") + a)), b.push(c)), 1;
  h = 0;
  d = "" === d ? "." : d + ":";
  if (I$1(a))
    for (var g = 0; g < a.length; g++) {
      k2 = a[g];
      var f2 = d + Q$1(k2, g);
      h += R$1(k2, b, e, f2, c);
    }
  else if (f2 = A$1(a), "function" === typeof f2)
    for (a = f2.call(a), g = 0; !(k2 = a.next()).done; )
      k2 = k2.value, f2 = d + Q$1(k2, g++), h += R$1(k2, b, e, f2, c);
  else if ("object" === k2)
    throw b = String(a), Error("Objects are not valid as a React child (found: " + ("[object Object]" === b ? "object with keys {" + Object.keys(a).join(", ") + "}" : b) + "). If you meant to render a collection of children, use an array instead.");
  return h;
}
function S$1(a, b, e) {
  if (null == a)
    return a;
  var d = [], c = 0;
  R$1(a, d, "", "", function(a2) {
    return b.call(e, a2, c++);
  });
  return d;
}
function T$1(a) {
  if (-1 === a._status) {
    var b = a._result;
    b = b();
    b.then(function(b2) {
      if (0 === a._status || -1 === a._status)
        a._status = 1, a._result = b2;
    }, function(b2) {
      if (0 === a._status || -1 === a._status)
        a._status = 2, a._result = b2;
    });
    -1 === a._status && (a._status = 0, a._result = b);
  }
  if (1 === a._status)
    return a._result.default;
  throw a._result;
}
var U$1 = { current: null }, V$1 = { transition: null }, W$1 = { ReactCurrentDispatcher: U$1, ReactCurrentBatchConfig: V$1, ReactCurrentOwner: K$1 };
function X$2() {
  throw Error("act(...) is not supported in production builds of React.");
}
react_production_min.Children = { map: S$1, forEach: function(a, b, e) {
  S$1(a, function() {
    b.apply(this, arguments);
  }, e);
}, count: function(a) {
  var b = 0;
  S$1(a, function() {
    b++;
  });
  return b;
}, toArray: function(a) {
  return S$1(a, function(a2) {
    return a2;
  }) || [];
}, only: function(a) {
  if (!O$1(a))
    throw Error("React.Children.only expected to receive a single React element child.");
  return a;
} };
react_production_min.Component = E$1;
react_production_min.Fragment = p$2;
react_production_min.Profiler = r$1;
react_production_min.PureComponent = G$1;
react_production_min.StrictMode = q$1;
react_production_min.Suspense = w;
react_production_min.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED = W$1;
react_production_min.act = X$2;
react_production_min.cloneElement = function(a, b, e) {
  if (null === a || void 0 === a)
    throw Error("React.cloneElement(...): The argument must be a React element, but you passed " + a + ".");
  var d = C$1({}, a.props), c = a.key, k2 = a.ref, h = a._owner;
  if (null != b) {
    void 0 !== b.ref && (k2 = b.ref, h = K$1.current);
    void 0 !== b.key && (c = "" + b.key);
    if (a.type && a.type.defaultProps)
      var g = a.type.defaultProps;
    for (f2 in b)
      J.call(b, f2) && !L$1.hasOwnProperty(f2) && (d[f2] = void 0 === b[f2] && void 0 !== g ? g[f2] : b[f2]);
  }
  var f2 = arguments.length - 2;
  if (1 === f2)
    d.children = e;
  else if (1 < f2) {
    g = Array(f2);
    for (var m2 = 0; m2 < f2; m2++)
      g[m2] = arguments[m2 + 2];
    d.children = g;
  }
  return { $$typeof: l$1, type: a.type, key: c, ref: k2, props: d, _owner: h };
};
react_production_min.createContext = function(a) {
  a = { $$typeof: u, _currentValue: a, _currentValue2: a, _threadCount: 0, Provider: null, Consumer: null, _defaultValue: null, _globalName: null };
  a.Provider = { $$typeof: t, _context: a };
  return a.Consumer = a;
};
react_production_min.createElement = M$1;
react_production_min.createFactory = function(a) {
  var b = M$1.bind(null, a);
  b.type = a;
  return b;
};
react_production_min.createRef = function() {
  return { current: null };
};
react_production_min.forwardRef = function(a) {
  return { $$typeof: v$1, render: a };
};
react_production_min.isValidElement = O$1;
react_production_min.lazy = function(a) {
  return { $$typeof: y, _payload: { _status: -1, _result: a }, _init: T$1 };
};
react_production_min.memo = function(a, b) {
  return { $$typeof: x, type: a, compare: void 0 === b ? null : b };
};
react_production_min.startTransition = function(a) {
  var b = V$1.transition;
  V$1.transition = {};
  try {
    a();
  } finally {
    V$1.transition = b;
  }
};
react_production_min.unstable_act = X$2;
react_production_min.useCallback = function(a, b) {
  return U$1.current.useCallback(a, b);
};
react_production_min.useContext = function(a) {
  return U$1.current.useContext(a);
};
react_production_min.useDebugValue = function() {
};
react_production_min.useDeferredValue = function(a) {
  return U$1.current.useDeferredValue(a);
};
react_production_min.useEffect = function(a, b) {
  return U$1.current.useEffect(a, b);
};
react_production_min.useId = function() {
  return U$1.current.useId();
};
react_production_min.useImperativeHandle = function(a, b, e) {
  return U$1.current.useImperativeHandle(a, b, e);
};
react_production_min.useInsertionEffect = function(a, b) {
  return U$1.current.useInsertionEffect(a, b);
};
react_production_min.useLayoutEffect = function(a, b) {
  return U$1.current.useLayoutEffect(a, b);
};
react_production_min.useMemo = function(a, b) {
  return U$1.current.useMemo(a, b);
};
react_production_min.useReducer = function(a, b, e) {
  return U$1.current.useReducer(a, b, e);
};
react_production_min.useRef = function(a) {
  return U$1.current.useRef(a);
};
react_production_min.useState = function(a) {
  return U$1.current.useState(a);
};
react_production_min.useSyncExternalStore = function(a, b, e) {
  return U$1.current.useSyncExternalStore(a, b, e);
};
react_production_min.useTransition = function() {
  return U$1.current.useTransition();
};
react_production_min.version = "18.3.1";
{
  react.exports = react_production_min;
}
var reactExports = react.exports;
const React = /* @__PURE__ */ getDefaultExportFromCjs(reactExports);
/**
 * @license React
 * react-jsx-runtime.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var f = reactExports, k = Symbol.for("react.element"), l = Symbol.for("react.fragment"), m$1 = Object.prototype.hasOwnProperty, n = f.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED.ReactCurrentOwner, p$1 = { key: true, ref: true, __self: true, __source: true };
function q(c, a, g) {
  var b, d = {}, e = null, h = null;
  void 0 !== g && (e = "" + g);
  void 0 !== a.key && (e = "" + a.key);
  void 0 !== a.ref && (h = a.ref);
  for (b in a)
    m$1.call(a, b) && !p$1.hasOwnProperty(b) && (d[b] = a[b]);
  if (c && c.defaultProps)
    for (b in a = c.defaultProps, a)
      void 0 === d[b] && (d[b] = a[b]);
  return { $$typeof: k, type: c, key: e, ref: h, props: d, _owner: n.current };
}
reactJsxRuntime_production_min.Fragment = l;
reactJsxRuntime_production_min.jsx = q;
reactJsxRuntime_production_min.jsxs = q;
{
  jsxRuntime.exports = reactJsxRuntime_production_min;
}
var jsxRuntimeExports = jsxRuntime.exports;
var client = {};
var reactDom = { exports: {} };
var reactDom_production_min = {};
var scheduler = { exports: {} };
var scheduler_production_min = {};
/**
 * @license React
 * scheduler.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
(function(exports) {
  function f2(a, b) {
    var c = a.length;
    a.push(b);
    a:
      for (; 0 < c; ) {
        var d = c - 1 >>> 1, e = a[d];
        if (0 < g(e, b))
          a[d] = b, a[c] = e, c = d;
        else
          break a;
      }
  }
  function h(a) {
    return 0 === a.length ? null : a[0];
  }
  function k2(a) {
    if (0 === a.length)
      return null;
    var b = a[0], c = a.pop();
    if (c !== b) {
      a[0] = c;
      a:
        for (var d = 0, e = a.length, w2 = e >>> 1; d < w2; ) {
          var m2 = 2 * (d + 1) - 1, C2 = a[m2], n2 = m2 + 1, x2 = a[n2];
          if (0 > g(C2, c))
            n2 < e && 0 > g(x2, C2) ? (a[d] = x2, a[n2] = c, d = n2) : (a[d] = C2, a[m2] = c, d = m2);
          else if (n2 < e && 0 > g(x2, c))
            a[d] = x2, a[n2] = c, d = n2;
          else
            break a;
        }
    }
    return b;
  }
  function g(a, b) {
    var c = a.sortIndex - b.sortIndex;
    return 0 !== c ? c : a.id - b.id;
  }
  if ("object" === typeof performance && "function" === typeof performance.now) {
    var l2 = performance;
    exports.unstable_now = function() {
      return l2.now();
    };
  } else {
    var p2 = Date, q2 = p2.now();
    exports.unstable_now = function() {
      return p2.now() - q2;
    };
  }
  var r2 = [], t2 = [], u2 = 1, v2 = null, y2 = 3, z2 = false, A2 = false, B2 = false, D2 = "function" === typeof setTimeout ? setTimeout : null, E2 = "function" === typeof clearTimeout ? clearTimeout : null, F2 = "undefined" !== typeof setImmediate ? setImmediate : null;
  "undefined" !== typeof navigator && void 0 !== navigator.scheduling && void 0 !== navigator.scheduling.isInputPending && navigator.scheduling.isInputPending.bind(navigator.scheduling);
  function G2(a) {
    for (var b = h(t2); null !== b; ) {
      if (null === b.callback)
        k2(t2);
      else if (b.startTime <= a)
        k2(t2), b.sortIndex = b.expirationTime, f2(r2, b);
      else
        break;
      b = h(t2);
    }
  }
  function H2(a) {
    B2 = false;
    G2(a);
    if (!A2)
      if (null !== h(r2))
        A2 = true, I2(J2);
      else {
        var b = h(t2);
        null !== b && K2(H2, b.startTime - a);
      }
  }
  function J2(a, b) {
    A2 = false;
    B2 && (B2 = false, E2(L2), L2 = -1);
    z2 = true;
    var c = y2;
    try {
      G2(b);
      for (v2 = h(r2); null !== v2 && (!(v2.expirationTime > b) || a && !M2()); ) {
        var d = v2.callback;
        if ("function" === typeof d) {
          v2.callback = null;
          y2 = v2.priorityLevel;
          var e = d(v2.expirationTime <= b);
          b = exports.unstable_now();
          "function" === typeof e ? v2.callback = e : v2 === h(r2) && k2(r2);
          G2(b);
        } else
          k2(r2);
        v2 = h(r2);
      }
      if (null !== v2)
        var w2 = true;
      else {
        var m2 = h(t2);
        null !== m2 && K2(H2, m2.startTime - b);
        w2 = false;
      }
      return w2;
    } finally {
      v2 = null, y2 = c, z2 = false;
    }
  }
  var N2 = false, O2 = null, L2 = -1, P2 = 5, Q2 = -1;
  function M2() {
    return exports.unstable_now() - Q2 < P2 ? false : true;
  }
  function R2() {
    if (null !== O2) {
      var a = exports.unstable_now();
      Q2 = a;
      var b = true;
      try {
        b = O2(true, a);
      } finally {
        b ? S2() : (N2 = false, O2 = null);
      }
    } else
      N2 = false;
  }
  var S2;
  if ("function" === typeof F2)
    S2 = function() {
      F2(R2);
    };
  else if ("undefined" !== typeof MessageChannel) {
    var T2 = new MessageChannel(), U2 = T2.port2;
    T2.port1.onmessage = R2;
    S2 = function() {
      U2.postMessage(null);
    };
  } else
    S2 = function() {
      D2(R2, 0);
    };
  function I2(a) {
    O2 = a;
    N2 || (N2 = true, S2());
  }
  function K2(a, b) {
    L2 = D2(function() {
      a(exports.unstable_now());
    }, b);
  }
  exports.unstable_IdlePriority = 5;
  exports.unstable_ImmediatePriority = 1;
  exports.unstable_LowPriority = 4;
  exports.unstable_NormalPriority = 3;
  exports.unstable_Profiling = null;
  exports.unstable_UserBlockingPriority = 2;
  exports.unstable_cancelCallback = function(a) {
    a.callback = null;
  };
  exports.unstable_continueExecution = function() {
    A2 || z2 || (A2 = true, I2(J2));
  };
  exports.unstable_forceFrameRate = function(a) {
    0 > a || 125 < a ? console.error("forceFrameRate takes a positive int between 0 and 125, forcing frame rates higher than 125 fps is not supported") : P2 = 0 < a ? Math.floor(1e3 / a) : 5;
  };
  exports.unstable_getCurrentPriorityLevel = function() {
    return y2;
  };
  exports.unstable_getFirstCallbackNode = function() {
    return h(r2);
  };
  exports.unstable_next = function(a) {
    switch (y2) {
      case 1:
      case 2:
      case 3:
        var b = 3;
        break;
      default:
        b = y2;
    }
    var c = y2;
    y2 = b;
    try {
      return a();
    } finally {
      y2 = c;
    }
  };
  exports.unstable_pauseExecution = function() {
  };
  exports.unstable_requestPaint = function() {
  };
  exports.unstable_runWithPriority = function(a, b) {
    switch (a) {
      case 1:
      case 2:
      case 3:
      case 4:
      case 5:
        break;
      default:
        a = 3;
    }
    var c = y2;
    y2 = a;
    try {
      return b();
    } finally {
      y2 = c;
    }
  };
  exports.unstable_scheduleCallback = function(a, b, c) {
    var d = exports.unstable_now();
    "object" === typeof c && null !== c ? (c = c.delay, c = "number" === typeof c && 0 < c ? d + c : d) : c = d;
    switch (a) {
      case 1:
        var e = -1;
        break;
      case 2:
        e = 250;
        break;
      case 5:
        e = 1073741823;
        break;
      case 4:
        e = 1e4;
        break;
      default:
        e = 5e3;
    }
    e = c + e;
    a = { id: u2++, callback: b, priorityLevel: a, startTime: c, expirationTime: e, sortIndex: -1 };
    c > d ? (a.sortIndex = c, f2(t2, a), null === h(r2) && a === h(t2) && (B2 ? (E2(L2), L2 = -1) : B2 = true, K2(H2, c - d))) : (a.sortIndex = e, f2(r2, a), A2 || z2 || (A2 = true, I2(J2)));
    return a;
  };
  exports.unstable_shouldYield = M2;
  exports.unstable_wrapCallback = function(a) {
    var b = y2;
    return function() {
      var c = y2;
      y2 = b;
      try {
        return a.apply(this, arguments);
      } finally {
        y2 = c;
      }
    };
  };
})(scheduler_production_min);
{
  scheduler.exports = scheduler_production_min;
}
var schedulerExports = scheduler.exports;
/**
 * @license React
 * react-dom.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var aa = reactExports, ca = schedulerExports;
function p(a) {
  for (var b = "https://reactjs.org/docs/error-decoder.html?invariant=" + a, c = 1; c < arguments.length; c++)
    b += "&args[]=" + encodeURIComponent(arguments[c]);
  return "Minified React error #" + a + "; visit " + b + " for the full message or use the non-minified dev environment for full errors and additional helpful warnings.";
}
var da = /* @__PURE__ */ new Set(), ea = {};
function fa(a, b) {
  ha(a, b);
  ha(a + "Capture", b);
}
function ha(a, b) {
  ea[a] = b;
  for (a = 0; a < b.length; a++)
    da.add(b[a]);
}
var ia = !("undefined" === typeof window || "undefined" === typeof window.document || "undefined" === typeof window.document.createElement), ja = Object.prototype.hasOwnProperty, ka = /^[:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD][:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD\-.0-9\u00B7\u0300-\u036F\u203F-\u2040]*$/, la = {}, ma = {};
function oa(a) {
  if (ja.call(ma, a))
    return true;
  if (ja.call(la, a))
    return false;
  if (ka.test(a))
    return ma[a] = true;
  la[a] = true;
  return false;
}
function pa(a, b, c, d) {
  if (null !== c && 0 === c.type)
    return false;
  switch (typeof b) {
    case "function":
    case "symbol":
      return true;
    case "boolean":
      if (d)
        return false;
      if (null !== c)
        return !c.acceptsBooleans;
      a = a.toLowerCase().slice(0, 5);
      return "data-" !== a && "aria-" !== a;
    default:
      return false;
  }
}
function qa(a, b, c, d) {
  if (null === b || "undefined" === typeof b || pa(a, b, c, d))
    return true;
  if (d)
    return false;
  if (null !== c)
    switch (c.type) {
      case 3:
        return !b;
      case 4:
        return false === b;
      case 5:
        return isNaN(b);
      case 6:
        return isNaN(b) || 1 > b;
    }
  return false;
}
function v(a, b, c, d, e, f2, g) {
  this.acceptsBooleans = 2 === b || 3 === b || 4 === b;
  this.attributeName = d;
  this.attributeNamespace = e;
  this.mustUseProperty = c;
  this.propertyName = a;
  this.type = b;
  this.sanitizeURL = f2;
  this.removeEmptyString = g;
}
var z = {};
"children dangerouslySetInnerHTML defaultValue defaultChecked innerHTML suppressContentEditableWarning suppressHydrationWarning style".split(" ").forEach(function(a) {
  z[a] = new v(a, 0, false, a, null, false, false);
});
[["acceptCharset", "accept-charset"], ["className", "class"], ["htmlFor", "for"], ["httpEquiv", "http-equiv"]].forEach(function(a) {
  var b = a[0];
  z[b] = new v(b, 1, false, a[1], null, false, false);
});
["contentEditable", "draggable", "spellCheck", "value"].forEach(function(a) {
  z[a] = new v(a, 2, false, a.toLowerCase(), null, false, false);
});
["autoReverse", "externalResourcesRequired", "focusable", "preserveAlpha"].forEach(function(a) {
  z[a] = new v(a, 2, false, a, null, false, false);
});
"allowFullScreen async autoFocus autoPlay controls default defer disabled disablePictureInPicture disableRemotePlayback formNoValidate hidden loop noModule noValidate open playsInline readOnly required reversed scoped seamless itemScope".split(" ").forEach(function(a) {
  z[a] = new v(a, 3, false, a.toLowerCase(), null, false, false);
});
["checked", "multiple", "muted", "selected"].forEach(function(a) {
  z[a] = new v(a, 3, true, a, null, false, false);
});
["capture", "download"].forEach(function(a) {
  z[a] = new v(a, 4, false, a, null, false, false);
});
["cols", "rows", "size", "span"].forEach(function(a) {
  z[a] = new v(a, 6, false, a, null, false, false);
});
["rowSpan", "start"].forEach(function(a) {
  z[a] = new v(a, 5, false, a.toLowerCase(), null, false, false);
});
var ra = /[\-:]([a-z])/g;
function sa(a) {
  return a[1].toUpperCase();
}
"accent-height alignment-baseline arabic-form baseline-shift cap-height clip-path clip-rule color-interpolation color-interpolation-filters color-profile color-rendering dominant-baseline enable-background fill-opacity fill-rule flood-color flood-opacity font-family font-size font-size-adjust font-stretch font-style font-variant font-weight glyph-name glyph-orientation-horizontal glyph-orientation-vertical horiz-adv-x horiz-origin-x image-rendering letter-spacing lighting-color marker-end marker-mid marker-start overline-position overline-thickness paint-order panose-1 pointer-events rendering-intent shape-rendering stop-color stop-opacity strikethrough-position strikethrough-thickness stroke-dasharray stroke-dashoffset stroke-linecap stroke-linejoin stroke-miterlimit stroke-opacity stroke-width text-anchor text-decoration text-rendering underline-position underline-thickness unicode-bidi unicode-range units-per-em v-alphabetic v-hanging v-ideographic v-mathematical vector-effect vert-adv-y vert-origin-x vert-origin-y word-spacing writing-mode xmlns:xlink x-height".split(" ").forEach(function(a) {
  var b = a.replace(
    ra,
    sa
  );
  z[b] = new v(b, 1, false, a, null, false, false);
});
"xlink:actuate xlink:arcrole xlink:role xlink:show xlink:title xlink:type".split(" ").forEach(function(a) {
  var b = a.replace(ra, sa);
  z[b] = new v(b, 1, false, a, "http://www.w3.org/1999/xlink", false, false);
});
["xml:base", "xml:lang", "xml:space"].forEach(function(a) {
  var b = a.replace(ra, sa);
  z[b] = new v(b, 1, false, a, "http://www.w3.org/XML/1998/namespace", false, false);
});
["tabIndex", "crossOrigin"].forEach(function(a) {
  z[a] = new v(a, 1, false, a.toLowerCase(), null, false, false);
});
z.xlinkHref = new v("xlinkHref", 1, false, "xlink:href", "http://www.w3.org/1999/xlink", true, false);
["src", "href", "action", "formAction"].forEach(function(a) {
  z[a] = new v(a, 1, false, a.toLowerCase(), null, true, true);
});
function ta(a, b, c, d) {
  var e = z.hasOwnProperty(b) ? z[b] : null;
  if (null !== e ? 0 !== e.type : d || !(2 < b.length) || "o" !== b[0] && "O" !== b[0] || "n" !== b[1] && "N" !== b[1])
    qa(b, c, e, d) && (c = null), d || null === e ? oa(b) && (null === c ? a.removeAttribute(b) : a.setAttribute(b, "" + c)) : e.mustUseProperty ? a[e.propertyName] = null === c ? 3 === e.type ? false : "" : c : (b = e.attributeName, d = e.attributeNamespace, null === c ? a.removeAttribute(b) : (e = e.type, c = 3 === e || 4 === e && true === c ? "" : "" + c, d ? a.setAttributeNS(d, b, c) : a.setAttribute(b, c)));
}
var ua = aa.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED, va = Symbol.for("react.element"), wa = Symbol.for("react.portal"), ya = Symbol.for("react.fragment"), za = Symbol.for("react.strict_mode"), Aa = Symbol.for("react.profiler"), Ba = Symbol.for("react.provider"), Ca = Symbol.for("react.context"), Da = Symbol.for("react.forward_ref"), Ea = Symbol.for("react.suspense"), Fa = Symbol.for("react.suspense_list"), Ga = Symbol.for("react.memo"), Ha = Symbol.for("react.lazy");
var Ia = Symbol.for("react.offscreen");
var Ja = Symbol.iterator;
function Ka(a) {
  if (null === a || "object" !== typeof a)
    return null;
  a = Ja && a[Ja] || a["@@iterator"];
  return "function" === typeof a ? a : null;
}
var A = Object.assign, La;
function Ma(a) {
  if (void 0 === La)
    try {
      throw Error();
    } catch (c) {
      var b = c.stack.trim().match(/\n( *(at )?)/);
      La = b && b[1] || "";
    }
  return "\n" + La + a;
}
var Na = false;
function Oa(a, b) {
  if (!a || Na)
    return "";
  Na = true;
  var c = Error.prepareStackTrace;
  Error.prepareStackTrace = void 0;
  try {
    if (b)
      if (b = function() {
        throw Error();
      }, Object.defineProperty(b.prototype, "props", { set: function() {
        throw Error();
      } }), "object" === typeof Reflect && Reflect.construct) {
        try {
          Reflect.construct(b, []);
        } catch (l2) {
          var d = l2;
        }
        Reflect.construct(a, [], b);
      } else {
        try {
          b.call();
        } catch (l2) {
          d = l2;
        }
        a.call(b.prototype);
      }
    else {
      try {
        throw Error();
      } catch (l2) {
        d = l2;
      }
      a();
    }
  } catch (l2) {
    if (l2 && d && "string" === typeof l2.stack) {
      for (var e = l2.stack.split("\n"), f2 = d.stack.split("\n"), g = e.length - 1, h = f2.length - 1; 1 <= g && 0 <= h && e[g] !== f2[h]; )
        h--;
      for (; 1 <= g && 0 <= h; g--, h--)
        if (e[g] !== f2[h]) {
          if (1 !== g || 1 !== h) {
            do
              if (g--, h--, 0 > h || e[g] !== f2[h]) {
                var k2 = "\n" + e[g].replace(" at new ", " at ");
                a.displayName && k2.includes("<anonymous>") && (k2 = k2.replace("<anonymous>", a.displayName));
                return k2;
              }
            while (1 <= g && 0 <= h);
          }
          break;
        }
    }
  } finally {
    Na = false, Error.prepareStackTrace = c;
  }
  return (a = a ? a.displayName || a.name : "") ? Ma(a) : "";
}
function Pa(a) {
  switch (a.tag) {
    case 5:
      return Ma(a.type);
    case 16:
      return Ma("Lazy");
    case 13:
      return Ma("Suspense");
    case 19:
      return Ma("SuspenseList");
    case 0:
    case 2:
    case 15:
      return a = Oa(a.type, false), a;
    case 11:
      return a = Oa(a.type.render, false), a;
    case 1:
      return a = Oa(a.type, true), a;
    default:
      return "";
  }
}
function Qa(a) {
  if (null == a)
    return null;
  if ("function" === typeof a)
    return a.displayName || a.name || null;
  if ("string" === typeof a)
    return a;
  switch (a) {
    case ya:
      return "Fragment";
    case wa:
      return "Portal";
    case Aa:
      return "Profiler";
    case za:
      return "StrictMode";
    case Ea:
      return "Suspense";
    case Fa:
      return "SuspenseList";
  }
  if ("object" === typeof a)
    switch (a.$$typeof) {
      case Ca:
        return (a.displayName || "Context") + ".Consumer";
      case Ba:
        return (a._context.displayName || "Context") + ".Provider";
      case Da:
        var b = a.render;
        a = a.displayName;
        a || (a = b.displayName || b.name || "", a = "" !== a ? "ForwardRef(" + a + ")" : "ForwardRef");
        return a;
      case Ga:
        return b = a.displayName || null, null !== b ? b : Qa(a.type) || "Memo";
      case Ha:
        b = a._payload;
        a = a._init;
        try {
          return Qa(a(b));
        } catch (c) {
        }
    }
  return null;
}
function Ra(a) {
  var b = a.type;
  switch (a.tag) {
    case 24:
      return "Cache";
    case 9:
      return (b.displayName || "Context") + ".Consumer";
    case 10:
      return (b._context.displayName || "Context") + ".Provider";
    case 18:
      return "DehydratedFragment";
    case 11:
      return a = b.render, a = a.displayName || a.name || "", b.displayName || ("" !== a ? "ForwardRef(" + a + ")" : "ForwardRef");
    case 7:
      return "Fragment";
    case 5:
      return b;
    case 4:
      return "Portal";
    case 3:
      return "Root";
    case 6:
      return "Text";
    case 16:
      return Qa(b);
    case 8:
      return b === za ? "StrictMode" : "Mode";
    case 22:
      return "Offscreen";
    case 12:
      return "Profiler";
    case 21:
      return "Scope";
    case 13:
      return "Suspense";
    case 19:
      return "SuspenseList";
    case 25:
      return "TracingMarker";
    case 1:
    case 0:
    case 17:
    case 2:
    case 14:
    case 15:
      if ("function" === typeof b)
        return b.displayName || b.name || null;
      if ("string" === typeof b)
        return b;
  }
  return null;
}
function Sa(a) {
  switch (typeof a) {
    case "boolean":
    case "number":
    case "string":
    case "undefined":
      return a;
    case "object":
      return a;
    default:
      return "";
  }
}
function Ta(a) {
  var b = a.type;
  return (a = a.nodeName) && "input" === a.toLowerCase() && ("checkbox" === b || "radio" === b);
}
function Ua(a) {
  var b = Ta(a) ? "checked" : "value", c = Object.getOwnPropertyDescriptor(a.constructor.prototype, b), d = "" + a[b];
  if (!a.hasOwnProperty(b) && "undefined" !== typeof c && "function" === typeof c.get && "function" === typeof c.set) {
    var e = c.get, f2 = c.set;
    Object.defineProperty(a, b, { configurable: true, get: function() {
      return e.call(this);
    }, set: function(a2) {
      d = "" + a2;
      f2.call(this, a2);
    } });
    Object.defineProperty(a, b, { enumerable: c.enumerable });
    return { getValue: function() {
      return d;
    }, setValue: function(a2) {
      d = "" + a2;
    }, stopTracking: function() {
      a._valueTracker = null;
      delete a[b];
    } };
  }
}
function Va(a) {
  a._valueTracker || (a._valueTracker = Ua(a));
}
function Wa(a) {
  if (!a)
    return false;
  var b = a._valueTracker;
  if (!b)
    return true;
  var c = b.getValue();
  var d = "";
  a && (d = Ta(a) ? a.checked ? "true" : "false" : a.value);
  a = d;
  return a !== c ? (b.setValue(a), true) : false;
}
function Xa(a) {
  a = a || ("undefined" !== typeof document ? document : void 0);
  if ("undefined" === typeof a)
    return null;
  try {
    return a.activeElement || a.body;
  } catch (b) {
    return a.body;
  }
}
function Ya(a, b) {
  var c = b.checked;
  return A({}, b, { defaultChecked: void 0, defaultValue: void 0, value: void 0, checked: null != c ? c : a._wrapperState.initialChecked });
}
function Za(a, b) {
  var c = null == b.defaultValue ? "" : b.defaultValue, d = null != b.checked ? b.checked : b.defaultChecked;
  c = Sa(null != b.value ? b.value : c);
  a._wrapperState = { initialChecked: d, initialValue: c, controlled: "checkbox" === b.type || "radio" === b.type ? null != b.checked : null != b.value };
}
function ab(a, b) {
  b = b.checked;
  null != b && ta(a, "checked", b, false);
}
function bb(a, b) {
  ab(a, b);
  var c = Sa(b.value), d = b.type;
  if (null != c)
    if ("number" === d) {
      if (0 === c && "" === a.value || a.value != c)
        a.value = "" + c;
    } else
      a.value !== "" + c && (a.value = "" + c);
  else if ("submit" === d || "reset" === d) {
    a.removeAttribute("value");
    return;
  }
  b.hasOwnProperty("value") ? cb(a, b.type, c) : b.hasOwnProperty("defaultValue") && cb(a, b.type, Sa(b.defaultValue));
  null == b.checked && null != b.defaultChecked && (a.defaultChecked = !!b.defaultChecked);
}
function db(a, b, c) {
  if (b.hasOwnProperty("value") || b.hasOwnProperty("defaultValue")) {
    var d = b.type;
    if (!("submit" !== d && "reset" !== d || void 0 !== b.value && null !== b.value))
      return;
    b = "" + a._wrapperState.initialValue;
    c || b === a.value || (a.value = b);
    a.defaultValue = b;
  }
  c = a.name;
  "" !== c && (a.name = "");
  a.defaultChecked = !!a._wrapperState.initialChecked;
  "" !== c && (a.name = c);
}
function cb(a, b, c) {
  if ("number" !== b || Xa(a.ownerDocument) !== a)
    null == c ? a.defaultValue = "" + a._wrapperState.initialValue : a.defaultValue !== "" + c && (a.defaultValue = "" + c);
}
var eb = Array.isArray;
function fb(a, b, c, d) {
  a = a.options;
  if (b) {
    b = {};
    for (var e = 0; e < c.length; e++)
      b["$" + c[e]] = true;
    for (c = 0; c < a.length; c++)
      e = b.hasOwnProperty("$" + a[c].value), a[c].selected !== e && (a[c].selected = e), e && d && (a[c].defaultSelected = true);
  } else {
    c = "" + Sa(c);
    b = null;
    for (e = 0; e < a.length; e++) {
      if (a[e].value === c) {
        a[e].selected = true;
        d && (a[e].defaultSelected = true);
        return;
      }
      null !== b || a[e].disabled || (b = a[e]);
    }
    null !== b && (b.selected = true);
  }
}
function gb(a, b) {
  if (null != b.dangerouslySetInnerHTML)
    throw Error(p(91));
  return A({}, b, { value: void 0, defaultValue: void 0, children: "" + a._wrapperState.initialValue });
}
function hb(a, b) {
  var c = b.value;
  if (null == c) {
    c = b.children;
    b = b.defaultValue;
    if (null != c) {
      if (null != b)
        throw Error(p(92));
      if (eb(c)) {
        if (1 < c.length)
          throw Error(p(93));
        c = c[0];
      }
      b = c;
    }
    null == b && (b = "");
    c = b;
  }
  a._wrapperState = { initialValue: Sa(c) };
}
function ib(a, b) {
  var c = Sa(b.value), d = Sa(b.defaultValue);
  null != c && (c = "" + c, c !== a.value && (a.value = c), null == b.defaultValue && a.defaultValue !== c && (a.defaultValue = c));
  null != d && (a.defaultValue = "" + d);
}
function jb(a) {
  var b = a.textContent;
  b === a._wrapperState.initialValue && "" !== b && null !== b && (a.value = b);
}
function kb(a) {
  switch (a) {
    case "svg":
      return "http://www.w3.org/2000/svg";
    case "math":
      return "http://www.w3.org/1998/Math/MathML";
    default:
      return "http://www.w3.org/1999/xhtml";
  }
}
function lb(a, b) {
  return null == a || "http://www.w3.org/1999/xhtml" === a ? kb(b) : "http://www.w3.org/2000/svg" === a && "foreignObject" === b ? "http://www.w3.org/1999/xhtml" : a;
}
var mb, nb = function(a) {
  return "undefined" !== typeof MSApp && MSApp.execUnsafeLocalFunction ? function(b, c, d, e) {
    MSApp.execUnsafeLocalFunction(function() {
      return a(b, c, d, e);
    });
  } : a;
}(function(a, b) {
  if ("http://www.w3.org/2000/svg" !== a.namespaceURI || "innerHTML" in a)
    a.innerHTML = b;
  else {
    mb = mb || document.createElement("div");
    mb.innerHTML = "<svg>" + b.valueOf().toString() + "</svg>";
    for (b = mb.firstChild; a.firstChild; )
      a.removeChild(a.firstChild);
    for (; b.firstChild; )
      a.appendChild(b.firstChild);
  }
});
function ob(a, b) {
  if (b) {
    var c = a.firstChild;
    if (c && c === a.lastChild && 3 === c.nodeType) {
      c.nodeValue = b;
      return;
    }
  }
  a.textContent = b;
}
var pb = {
  animationIterationCount: true,
  aspectRatio: true,
  borderImageOutset: true,
  borderImageSlice: true,
  borderImageWidth: true,
  boxFlex: true,
  boxFlexGroup: true,
  boxOrdinalGroup: true,
  columnCount: true,
  columns: true,
  flex: true,
  flexGrow: true,
  flexPositive: true,
  flexShrink: true,
  flexNegative: true,
  flexOrder: true,
  gridArea: true,
  gridRow: true,
  gridRowEnd: true,
  gridRowSpan: true,
  gridRowStart: true,
  gridColumn: true,
  gridColumnEnd: true,
  gridColumnSpan: true,
  gridColumnStart: true,
  fontWeight: true,
  lineClamp: true,
  lineHeight: true,
  opacity: true,
  order: true,
  orphans: true,
  tabSize: true,
  widows: true,
  zIndex: true,
  zoom: true,
  fillOpacity: true,
  floodOpacity: true,
  stopOpacity: true,
  strokeDasharray: true,
  strokeDashoffset: true,
  strokeMiterlimit: true,
  strokeOpacity: true,
  strokeWidth: true
}, qb = ["Webkit", "ms", "Moz", "O"];
Object.keys(pb).forEach(function(a) {
  qb.forEach(function(b) {
    b = b + a.charAt(0).toUpperCase() + a.substring(1);
    pb[b] = pb[a];
  });
});
function rb(a, b, c) {
  return null == b || "boolean" === typeof b || "" === b ? "" : c || "number" !== typeof b || 0 === b || pb.hasOwnProperty(a) && pb[a] ? ("" + b).trim() : b + "px";
}
function sb(a, b) {
  a = a.style;
  for (var c in b)
    if (b.hasOwnProperty(c)) {
      var d = 0 === c.indexOf("--"), e = rb(c, b[c], d);
      "float" === c && (c = "cssFloat");
      d ? a.setProperty(c, e) : a[c] = e;
    }
}
var tb = A({ menuitem: true }, { area: true, base: true, br: true, col: true, embed: true, hr: true, img: true, input: true, keygen: true, link: true, meta: true, param: true, source: true, track: true, wbr: true });
function ub(a, b) {
  if (b) {
    if (tb[a] && (null != b.children || null != b.dangerouslySetInnerHTML))
      throw Error(p(137, a));
    if (null != b.dangerouslySetInnerHTML) {
      if (null != b.children)
        throw Error(p(60));
      if ("object" !== typeof b.dangerouslySetInnerHTML || !("__html" in b.dangerouslySetInnerHTML))
        throw Error(p(61));
    }
    if (null != b.style && "object" !== typeof b.style)
      throw Error(p(62));
  }
}
function vb(a, b) {
  if (-1 === a.indexOf("-"))
    return "string" === typeof b.is;
  switch (a) {
    case "annotation-xml":
    case "color-profile":
    case "font-face":
    case "font-face-src":
    case "font-face-uri":
    case "font-face-format":
    case "font-face-name":
    case "missing-glyph":
      return false;
    default:
      return true;
  }
}
var wb = null;
function xb(a) {
  a = a.target || a.srcElement || window;
  a.correspondingUseElement && (a = a.correspondingUseElement);
  return 3 === a.nodeType ? a.parentNode : a;
}
var yb = null, zb = null, Ab = null;
function Bb(a) {
  if (a = Cb(a)) {
    if ("function" !== typeof yb)
      throw Error(p(280));
    var b = a.stateNode;
    b && (b = Db(b), yb(a.stateNode, a.type, b));
  }
}
function Eb(a) {
  zb ? Ab ? Ab.push(a) : Ab = [a] : zb = a;
}
function Fb() {
  if (zb) {
    var a = zb, b = Ab;
    Ab = zb = null;
    Bb(a);
    if (b)
      for (a = 0; a < b.length; a++)
        Bb(b[a]);
  }
}
function Gb(a, b) {
  return a(b);
}
function Hb() {
}
var Ib = false;
function Jb(a, b, c) {
  if (Ib)
    return a(b, c);
  Ib = true;
  try {
    return Gb(a, b, c);
  } finally {
    if (Ib = false, null !== zb || null !== Ab)
      Hb(), Fb();
  }
}
function Kb(a, b) {
  var c = a.stateNode;
  if (null === c)
    return null;
  var d = Db(c);
  if (null === d)
    return null;
  c = d[b];
  a:
    switch (b) {
      case "onClick":
      case "onClickCapture":
      case "onDoubleClick":
      case "onDoubleClickCapture":
      case "onMouseDown":
      case "onMouseDownCapture":
      case "onMouseMove":
      case "onMouseMoveCapture":
      case "onMouseUp":
      case "onMouseUpCapture":
      case "onMouseEnter":
        (d = !d.disabled) || (a = a.type, d = !("button" === a || "input" === a || "select" === a || "textarea" === a));
        a = !d;
        break a;
      default:
        a = false;
    }
  if (a)
    return null;
  if (c && "function" !== typeof c)
    throw Error(p(231, b, typeof c));
  return c;
}
var Lb = false;
if (ia)
  try {
    var Mb = {};
    Object.defineProperty(Mb, "passive", { get: function() {
      Lb = true;
    } });
    window.addEventListener("test", Mb, Mb);
    window.removeEventListener("test", Mb, Mb);
  } catch (a) {
    Lb = false;
  }
function Nb(a, b, c, d, e, f2, g, h, k2) {
  var l2 = Array.prototype.slice.call(arguments, 3);
  try {
    b.apply(c, l2);
  } catch (m2) {
    this.onError(m2);
  }
}
var Ob = false, Pb = null, Qb = false, Rb = null, Sb = { onError: function(a) {
  Ob = true;
  Pb = a;
} };
function Tb(a, b, c, d, e, f2, g, h, k2) {
  Ob = false;
  Pb = null;
  Nb.apply(Sb, arguments);
}
function Ub(a, b, c, d, e, f2, g, h, k2) {
  Tb.apply(this, arguments);
  if (Ob) {
    if (Ob) {
      var l2 = Pb;
      Ob = false;
      Pb = null;
    } else
      throw Error(p(198));
    Qb || (Qb = true, Rb = l2);
  }
}
function Vb(a) {
  var b = a, c = a;
  if (a.alternate)
    for (; b.return; )
      b = b.return;
  else {
    a = b;
    do
      b = a, 0 !== (b.flags & 4098) && (c = b.return), a = b.return;
    while (a);
  }
  return 3 === b.tag ? c : null;
}
function Wb(a) {
  if (13 === a.tag) {
    var b = a.memoizedState;
    null === b && (a = a.alternate, null !== a && (b = a.memoizedState));
    if (null !== b)
      return b.dehydrated;
  }
  return null;
}
function Xb(a) {
  if (Vb(a) !== a)
    throw Error(p(188));
}
function Yb(a) {
  var b = a.alternate;
  if (!b) {
    b = Vb(a);
    if (null === b)
      throw Error(p(188));
    return b !== a ? null : a;
  }
  for (var c = a, d = b; ; ) {
    var e = c.return;
    if (null === e)
      break;
    var f2 = e.alternate;
    if (null === f2) {
      d = e.return;
      if (null !== d) {
        c = d;
        continue;
      }
      break;
    }
    if (e.child === f2.child) {
      for (f2 = e.child; f2; ) {
        if (f2 === c)
          return Xb(e), a;
        if (f2 === d)
          return Xb(e), b;
        f2 = f2.sibling;
      }
      throw Error(p(188));
    }
    if (c.return !== d.return)
      c = e, d = f2;
    else {
      for (var g = false, h = e.child; h; ) {
        if (h === c) {
          g = true;
          c = e;
          d = f2;
          break;
        }
        if (h === d) {
          g = true;
          d = e;
          c = f2;
          break;
        }
        h = h.sibling;
      }
      if (!g) {
        for (h = f2.child; h; ) {
          if (h === c) {
            g = true;
            c = f2;
            d = e;
            break;
          }
          if (h === d) {
            g = true;
            d = f2;
            c = e;
            break;
          }
          h = h.sibling;
        }
        if (!g)
          throw Error(p(189));
      }
    }
    if (c.alternate !== d)
      throw Error(p(190));
  }
  if (3 !== c.tag)
    throw Error(p(188));
  return c.stateNode.current === c ? a : b;
}
function Zb(a) {
  a = Yb(a);
  return null !== a ? $b(a) : null;
}
function $b(a) {
  if (5 === a.tag || 6 === a.tag)
    return a;
  for (a = a.child; null !== a; ) {
    var b = $b(a);
    if (null !== b)
      return b;
    a = a.sibling;
  }
  return null;
}
var ac = ca.unstable_scheduleCallback, bc = ca.unstable_cancelCallback, cc = ca.unstable_shouldYield, dc = ca.unstable_requestPaint, B = ca.unstable_now, ec = ca.unstable_getCurrentPriorityLevel, fc = ca.unstable_ImmediatePriority, gc = ca.unstable_UserBlockingPriority, hc = ca.unstable_NormalPriority, ic = ca.unstable_LowPriority, jc = ca.unstable_IdlePriority, kc = null, lc = null;
function mc(a) {
  if (lc && "function" === typeof lc.onCommitFiberRoot)
    try {
      lc.onCommitFiberRoot(kc, a, void 0, 128 === (a.current.flags & 128));
    } catch (b) {
    }
}
var oc = Math.clz32 ? Math.clz32 : nc, pc = Math.log, qc = Math.LN2;
function nc(a) {
  a >>>= 0;
  return 0 === a ? 32 : 31 - (pc(a) / qc | 0) | 0;
}
var rc = 64, sc = 4194304;
function tc(a) {
  switch (a & -a) {
    case 1:
      return 1;
    case 2:
      return 2;
    case 4:
      return 4;
    case 8:
      return 8;
    case 16:
      return 16;
    case 32:
      return 32;
    case 64:
    case 128:
    case 256:
    case 512:
    case 1024:
    case 2048:
    case 4096:
    case 8192:
    case 16384:
    case 32768:
    case 65536:
    case 131072:
    case 262144:
    case 524288:
    case 1048576:
    case 2097152:
      return a & 4194240;
    case 4194304:
    case 8388608:
    case 16777216:
    case 33554432:
    case 67108864:
      return a & 130023424;
    case 134217728:
      return 134217728;
    case 268435456:
      return 268435456;
    case 536870912:
      return 536870912;
    case 1073741824:
      return 1073741824;
    default:
      return a;
  }
}
function uc(a, b) {
  var c = a.pendingLanes;
  if (0 === c)
    return 0;
  var d = 0, e = a.suspendedLanes, f2 = a.pingedLanes, g = c & 268435455;
  if (0 !== g) {
    var h = g & ~e;
    0 !== h ? d = tc(h) : (f2 &= g, 0 !== f2 && (d = tc(f2)));
  } else
    g = c & ~e, 0 !== g ? d = tc(g) : 0 !== f2 && (d = tc(f2));
  if (0 === d)
    return 0;
  if (0 !== b && b !== d && 0 === (b & e) && (e = d & -d, f2 = b & -b, e >= f2 || 16 === e && 0 !== (f2 & 4194240)))
    return b;
  0 !== (d & 4) && (d |= c & 16);
  b = a.entangledLanes;
  if (0 !== b)
    for (a = a.entanglements, b &= d; 0 < b; )
      c = 31 - oc(b), e = 1 << c, d |= a[c], b &= ~e;
  return d;
}
function vc(a, b) {
  switch (a) {
    case 1:
    case 2:
    case 4:
      return b + 250;
    case 8:
    case 16:
    case 32:
    case 64:
    case 128:
    case 256:
    case 512:
    case 1024:
    case 2048:
    case 4096:
    case 8192:
    case 16384:
    case 32768:
    case 65536:
    case 131072:
    case 262144:
    case 524288:
    case 1048576:
    case 2097152:
      return b + 5e3;
    case 4194304:
    case 8388608:
    case 16777216:
    case 33554432:
    case 67108864:
      return -1;
    case 134217728:
    case 268435456:
    case 536870912:
    case 1073741824:
      return -1;
    default:
      return -1;
  }
}
function wc(a, b) {
  for (var c = a.suspendedLanes, d = a.pingedLanes, e = a.expirationTimes, f2 = a.pendingLanes; 0 < f2; ) {
    var g = 31 - oc(f2), h = 1 << g, k2 = e[g];
    if (-1 === k2) {
      if (0 === (h & c) || 0 !== (h & d))
        e[g] = vc(h, b);
    } else
      k2 <= b && (a.expiredLanes |= h);
    f2 &= ~h;
  }
}
function xc(a) {
  a = a.pendingLanes & -1073741825;
  return 0 !== a ? a : a & 1073741824 ? 1073741824 : 0;
}
function yc() {
  var a = rc;
  rc <<= 1;
  0 === (rc & 4194240) && (rc = 64);
  return a;
}
function zc(a) {
  for (var b = [], c = 0; 31 > c; c++)
    b.push(a);
  return b;
}
function Ac(a, b, c) {
  a.pendingLanes |= b;
  536870912 !== b && (a.suspendedLanes = 0, a.pingedLanes = 0);
  a = a.eventTimes;
  b = 31 - oc(b);
  a[b] = c;
}
function Bc(a, b) {
  var c = a.pendingLanes & ~b;
  a.pendingLanes = b;
  a.suspendedLanes = 0;
  a.pingedLanes = 0;
  a.expiredLanes &= b;
  a.mutableReadLanes &= b;
  a.entangledLanes &= b;
  b = a.entanglements;
  var d = a.eventTimes;
  for (a = a.expirationTimes; 0 < c; ) {
    var e = 31 - oc(c), f2 = 1 << e;
    b[e] = 0;
    d[e] = -1;
    a[e] = -1;
    c &= ~f2;
  }
}
function Cc(a, b) {
  var c = a.entangledLanes |= b;
  for (a = a.entanglements; c; ) {
    var d = 31 - oc(c), e = 1 << d;
    e & b | a[d] & b && (a[d] |= b);
    c &= ~e;
  }
}
var C = 0;
function Dc(a) {
  a &= -a;
  return 1 < a ? 4 < a ? 0 !== (a & 268435455) ? 16 : 536870912 : 4 : 1;
}
var Ec, Fc, Gc, Hc, Ic, Jc = false, Kc = [], Lc = null, Mc = null, Nc = null, Oc = /* @__PURE__ */ new Map(), Pc = /* @__PURE__ */ new Map(), Qc = [], Rc = "mousedown mouseup touchcancel touchend touchstart auxclick dblclick pointercancel pointerdown pointerup dragend dragstart drop compositionend compositionstart keydown keypress keyup input textInput copy cut paste click change contextmenu reset submit".split(" ");
function Sc(a, b) {
  switch (a) {
    case "focusin":
    case "focusout":
      Lc = null;
      break;
    case "dragenter":
    case "dragleave":
      Mc = null;
      break;
    case "mouseover":
    case "mouseout":
      Nc = null;
      break;
    case "pointerover":
    case "pointerout":
      Oc.delete(b.pointerId);
      break;
    case "gotpointercapture":
    case "lostpointercapture":
      Pc.delete(b.pointerId);
  }
}
function Tc(a, b, c, d, e, f2) {
  if (null === a || a.nativeEvent !== f2)
    return a = { blockedOn: b, domEventName: c, eventSystemFlags: d, nativeEvent: f2, targetContainers: [e] }, null !== b && (b = Cb(b), null !== b && Fc(b)), a;
  a.eventSystemFlags |= d;
  b = a.targetContainers;
  null !== e && -1 === b.indexOf(e) && b.push(e);
  return a;
}
function Uc(a, b, c, d, e) {
  switch (b) {
    case "focusin":
      return Lc = Tc(Lc, a, b, c, d, e), true;
    case "dragenter":
      return Mc = Tc(Mc, a, b, c, d, e), true;
    case "mouseover":
      return Nc = Tc(Nc, a, b, c, d, e), true;
    case "pointerover":
      var f2 = e.pointerId;
      Oc.set(f2, Tc(Oc.get(f2) || null, a, b, c, d, e));
      return true;
    case "gotpointercapture":
      return f2 = e.pointerId, Pc.set(f2, Tc(Pc.get(f2) || null, a, b, c, d, e)), true;
  }
  return false;
}
function Vc(a) {
  var b = Wc(a.target);
  if (null !== b) {
    var c = Vb(b);
    if (null !== c) {
      if (b = c.tag, 13 === b) {
        if (b = Wb(c), null !== b) {
          a.blockedOn = b;
          Ic(a.priority, function() {
            Gc(c);
          });
          return;
        }
      } else if (3 === b && c.stateNode.current.memoizedState.isDehydrated) {
        a.blockedOn = 3 === c.tag ? c.stateNode.containerInfo : null;
        return;
      }
    }
  }
  a.blockedOn = null;
}
function Xc(a) {
  if (null !== a.blockedOn)
    return false;
  for (var b = a.targetContainers; 0 < b.length; ) {
    var c = Yc(a.domEventName, a.eventSystemFlags, b[0], a.nativeEvent);
    if (null === c) {
      c = a.nativeEvent;
      var d = new c.constructor(c.type, c);
      wb = d;
      c.target.dispatchEvent(d);
      wb = null;
    } else
      return b = Cb(c), null !== b && Fc(b), a.blockedOn = c, false;
    b.shift();
  }
  return true;
}
function Zc(a, b, c) {
  Xc(a) && c.delete(b);
}
function $c() {
  Jc = false;
  null !== Lc && Xc(Lc) && (Lc = null);
  null !== Mc && Xc(Mc) && (Mc = null);
  null !== Nc && Xc(Nc) && (Nc = null);
  Oc.forEach(Zc);
  Pc.forEach(Zc);
}
function ad(a, b) {
  a.blockedOn === b && (a.blockedOn = null, Jc || (Jc = true, ca.unstable_scheduleCallback(ca.unstable_NormalPriority, $c)));
}
function bd(a) {
  function b(b2) {
    return ad(b2, a);
  }
  if (0 < Kc.length) {
    ad(Kc[0], a);
    for (var c = 1; c < Kc.length; c++) {
      var d = Kc[c];
      d.blockedOn === a && (d.blockedOn = null);
    }
  }
  null !== Lc && ad(Lc, a);
  null !== Mc && ad(Mc, a);
  null !== Nc && ad(Nc, a);
  Oc.forEach(b);
  Pc.forEach(b);
  for (c = 0; c < Qc.length; c++)
    d = Qc[c], d.blockedOn === a && (d.blockedOn = null);
  for (; 0 < Qc.length && (c = Qc[0], null === c.blockedOn); )
    Vc(c), null === c.blockedOn && Qc.shift();
}
var cd = ua.ReactCurrentBatchConfig, dd = true;
function ed(a, b, c, d) {
  var e = C, f2 = cd.transition;
  cd.transition = null;
  try {
    C = 1, fd(a, b, c, d);
  } finally {
    C = e, cd.transition = f2;
  }
}
function gd(a, b, c, d) {
  var e = C, f2 = cd.transition;
  cd.transition = null;
  try {
    C = 4, fd(a, b, c, d);
  } finally {
    C = e, cd.transition = f2;
  }
}
function fd(a, b, c, d) {
  if (dd) {
    var e = Yc(a, b, c, d);
    if (null === e)
      hd(a, b, d, id, c), Sc(a, d);
    else if (Uc(e, a, b, c, d))
      d.stopPropagation();
    else if (Sc(a, d), b & 4 && -1 < Rc.indexOf(a)) {
      for (; null !== e; ) {
        var f2 = Cb(e);
        null !== f2 && Ec(f2);
        f2 = Yc(a, b, c, d);
        null === f2 && hd(a, b, d, id, c);
        if (f2 === e)
          break;
        e = f2;
      }
      null !== e && d.stopPropagation();
    } else
      hd(a, b, d, null, c);
  }
}
var id = null;
function Yc(a, b, c, d) {
  id = null;
  a = xb(d);
  a = Wc(a);
  if (null !== a)
    if (b = Vb(a), null === b)
      a = null;
    else if (c = b.tag, 13 === c) {
      a = Wb(b);
      if (null !== a)
        return a;
      a = null;
    } else if (3 === c) {
      if (b.stateNode.current.memoizedState.isDehydrated)
        return 3 === b.tag ? b.stateNode.containerInfo : null;
      a = null;
    } else
      b !== a && (a = null);
  id = a;
  return null;
}
function jd(a) {
  switch (a) {
    case "cancel":
    case "click":
    case "close":
    case "contextmenu":
    case "copy":
    case "cut":
    case "auxclick":
    case "dblclick":
    case "dragend":
    case "dragstart":
    case "drop":
    case "focusin":
    case "focusout":
    case "input":
    case "invalid":
    case "keydown":
    case "keypress":
    case "keyup":
    case "mousedown":
    case "mouseup":
    case "paste":
    case "pause":
    case "play":
    case "pointercancel":
    case "pointerdown":
    case "pointerup":
    case "ratechange":
    case "reset":
    case "resize":
    case "seeked":
    case "submit":
    case "touchcancel":
    case "touchend":
    case "touchstart":
    case "volumechange":
    case "change":
    case "selectionchange":
    case "textInput":
    case "compositionstart":
    case "compositionend":
    case "compositionupdate":
    case "beforeblur":
    case "afterblur":
    case "beforeinput":
    case "blur":
    case "fullscreenchange":
    case "focus":
    case "hashchange":
    case "popstate":
    case "select":
    case "selectstart":
      return 1;
    case "drag":
    case "dragenter":
    case "dragexit":
    case "dragleave":
    case "dragover":
    case "mousemove":
    case "mouseout":
    case "mouseover":
    case "pointermove":
    case "pointerout":
    case "pointerover":
    case "scroll":
    case "toggle":
    case "touchmove":
    case "wheel":
    case "mouseenter":
    case "mouseleave":
    case "pointerenter":
    case "pointerleave":
      return 4;
    case "message":
      switch (ec()) {
        case fc:
          return 1;
        case gc:
          return 4;
        case hc:
        case ic:
          return 16;
        case jc:
          return 536870912;
        default:
          return 16;
      }
    default:
      return 16;
  }
}
var kd = null, ld = null, md = null;
function nd() {
  if (md)
    return md;
  var a, b = ld, c = b.length, d, e = "value" in kd ? kd.value : kd.textContent, f2 = e.length;
  for (a = 0; a < c && b[a] === e[a]; a++)
    ;
  var g = c - a;
  for (d = 1; d <= g && b[c - d] === e[f2 - d]; d++)
    ;
  return md = e.slice(a, 1 < d ? 1 - d : void 0);
}
function od(a) {
  var b = a.keyCode;
  "charCode" in a ? (a = a.charCode, 0 === a && 13 === b && (a = 13)) : a = b;
  10 === a && (a = 13);
  return 32 <= a || 13 === a ? a : 0;
}
function pd() {
  return true;
}
function qd() {
  return false;
}
function rd(a) {
  function b(b2, d, e, f2, g) {
    this._reactName = b2;
    this._targetInst = e;
    this.type = d;
    this.nativeEvent = f2;
    this.target = g;
    this.currentTarget = null;
    for (var c in a)
      a.hasOwnProperty(c) && (b2 = a[c], this[c] = b2 ? b2(f2) : f2[c]);
    this.isDefaultPrevented = (null != f2.defaultPrevented ? f2.defaultPrevented : false === f2.returnValue) ? pd : qd;
    this.isPropagationStopped = qd;
    return this;
  }
  A(b.prototype, { preventDefault: function() {
    this.defaultPrevented = true;
    var a2 = this.nativeEvent;
    a2 && (a2.preventDefault ? a2.preventDefault() : "unknown" !== typeof a2.returnValue && (a2.returnValue = false), this.isDefaultPrevented = pd);
  }, stopPropagation: function() {
    var a2 = this.nativeEvent;
    a2 && (a2.stopPropagation ? a2.stopPropagation() : "unknown" !== typeof a2.cancelBubble && (a2.cancelBubble = true), this.isPropagationStopped = pd);
  }, persist: function() {
  }, isPersistent: pd });
  return b;
}
var sd = { eventPhase: 0, bubbles: 0, cancelable: 0, timeStamp: function(a) {
  return a.timeStamp || Date.now();
}, defaultPrevented: 0, isTrusted: 0 }, td = rd(sd), ud = A({}, sd, { view: 0, detail: 0 }), vd = rd(ud), wd, xd, yd, Ad = A({}, ud, { screenX: 0, screenY: 0, clientX: 0, clientY: 0, pageX: 0, pageY: 0, ctrlKey: 0, shiftKey: 0, altKey: 0, metaKey: 0, getModifierState: zd, button: 0, buttons: 0, relatedTarget: function(a) {
  return void 0 === a.relatedTarget ? a.fromElement === a.srcElement ? a.toElement : a.fromElement : a.relatedTarget;
}, movementX: function(a) {
  if ("movementX" in a)
    return a.movementX;
  a !== yd && (yd && "mousemove" === a.type ? (wd = a.screenX - yd.screenX, xd = a.screenY - yd.screenY) : xd = wd = 0, yd = a);
  return wd;
}, movementY: function(a) {
  return "movementY" in a ? a.movementY : xd;
} }), Bd = rd(Ad), Cd = A({}, Ad, { dataTransfer: 0 }), Dd = rd(Cd), Ed = A({}, ud, { relatedTarget: 0 }), Fd = rd(Ed), Gd = A({}, sd, { animationName: 0, elapsedTime: 0, pseudoElement: 0 }), Hd = rd(Gd), Id = A({}, sd, { clipboardData: function(a) {
  return "clipboardData" in a ? a.clipboardData : window.clipboardData;
} }), Jd = rd(Id), Kd = A({}, sd, { data: 0 }), Ld = rd(Kd), Md = {
  Esc: "Escape",
  Spacebar: " ",
  Left: "ArrowLeft",
  Up: "ArrowUp",
  Right: "ArrowRight",
  Down: "ArrowDown",
  Del: "Delete",
  Win: "OS",
  Menu: "ContextMenu",
  Apps: "ContextMenu",
  Scroll: "ScrollLock",
  MozPrintableKey: "Unidentified"
}, Nd = {
  8: "Backspace",
  9: "Tab",
  12: "Clear",
  13: "Enter",
  16: "Shift",
  17: "Control",
  18: "Alt",
  19: "Pause",
  20: "CapsLock",
  27: "Escape",
  32: " ",
  33: "PageUp",
  34: "PageDown",
  35: "End",
  36: "Home",
  37: "ArrowLeft",
  38: "ArrowUp",
  39: "ArrowRight",
  40: "ArrowDown",
  45: "Insert",
  46: "Delete",
  112: "F1",
  113: "F2",
  114: "F3",
  115: "F4",
  116: "F5",
  117: "F6",
  118: "F7",
  119: "F8",
  120: "F9",
  121: "F10",
  122: "F11",
  123: "F12",
  144: "NumLock",
  145: "ScrollLock",
  224: "Meta"
}, Od = { Alt: "altKey", Control: "ctrlKey", Meta: "metaKey", Shift: "shiftKey" };
function Pd(a) {
  var b = this.nativeEvent;
  return b.getModifierState ? b.getModifierState(a) : (a = Od[a]) ? !!b[a] : false;
}
function zd() {
  return Pd;
}
var Qd = A({}, ud, { key: function(a) {
  if (a.key) {
    var b = Md[a.key] || a.key;
    if ("Unidentified" !== b)
      return b;
  }
  return "keypress" === a.type ? (a = od(a), 13 === a ? "Enter" : String.fromCharCode(a)) : "keydown" === a.type || "keyup" === a.type ? Nd[a.keyCode] || "Unidentified" : "";
}, code: 0, location: 0, ctrlKey: 0, shiftKey: 0, altKey: 0, metaKey: 0, repeat: 0, locale: 0, getModifierState: zd, charCode: function(a) {
  return "keypress" === a.type ? od(a) : 0;
}, keyCode: function(a) {
  return "keydown" === a.type || "keyup" === a.type ? a.keyCode : 0;
}, which: function(a) {
  return "keypress" === a.type ? od(a) : "keydown" === a.type || "keyup" === a.type ? a.keyCode : 0;
} }), Rd = rd(Qd), Sd = A({}, Ad, { pointerId: 0, width: 0, height: 0, pressure: 0, tangentialPressure: 0, tiltX: 0, tiltY: 0, twist: 0, pointerType: 0, isPrimary: 0 }), Td = rd(Sd), Ud = A({}, ud, { touches: 0, targetTouches: 0, changedTouches: 0, altKey: 0, metaKey: 0, ctrlKey: 0, shiftKey: 0, getModifierState: zd }), Vd = rd(Ud), Wd = A({}, sd, { propertyName: 0, elapsedTime: 0, pseudoElement: 0 }), Xd = rd(Wd), Yd = A({}, Ad, {
  deltaX: function(a) {
    return "deltaX" in a ? a.deltaX : "wheelDeltaX" in a ? -a.wheelDeltaX : 0;
  },
  deltaY: function(a) {
    return "deltaY" in a ? a.deltaY : "wheelDeltaY" in a ? -a.wheelDeltaY : "wheelDelta" in a ? -a.wheelDelta : 0;
  },
  deltaZ: 0,
  deltaMode: 0
}), Zd = rd(Yd), $d = [9, 13, 27, 32], ae = ia && "CompositionEvent" in window, be = null;
ia && "documentMode" in document && (be = document.documentMode);
var ce = ia && "TextEvent" in window && !be, de = ia && (!ae || be && 8 < be && 11 >= be), ee = String.fromCharCode(32), fe = false;
function ge(a, b) {
  switch (a) {
    case "keyup":
      return -1 !== $d.indexOf(b.keyCode);
    case "keydown":
      return 229 !== b.keyCode;
    case "keypress":
    case "mousedown":
    case "focusout":
      return true;
    default:
      return false;
  }
}
function he(a) {
  a = a.detail;
  return "object" === typeof a && "data" in a ? a.data : null;
}
var ie = false;
function je(a, b) {
  switch (a) {
    case "compositionend":
      return he(b);
    case "keypress":
      if (32 !== b.which)
        return null;
      fe = true;
      return ee;
    case "textInput":
      return a = b.data, a === ee && fe ? null : a;
    default:
      return null;
  }
}
function ke(a, b) {
  if (ie)
    return "compositionend" === a || !ae && ge(a, b) ? (a = nd(), md = ld = kd = null, ie = false, a) : null;
  switch (a) {
    case "paste":
      return null;
    case "keypress":
      if (!(b.ctrlKey || b.altKey || b.metaKey) || b.ctrlKey && b.altKey) {
        if (b.char && 1 < b.char.length)
          return b.char;
        if (b.which)
          return String.fromCharCode(b.which);
      }
      return null;
    case "compositionend":
      return de && "ko" !== b.locale ? null : b.data;
    default:
      return null;
  }
}
var le = { color: true, date: true, datetime: true, "datetime-local": true, email: true, month: true, number: true, password: true, range: true, search: true, tel: true, text: true, time: true, url: true, week: true };
function me(a) {
  var b = a && a.nodeName && a.nodeName.toLowerCase();
  return "input" === b ? !!le[a.type] : "textarea" === b ? true : false;
}
function ne(a, b, c, d) {
  Eb(d);
  b = oe(b, "onChange");
  0 < b.length && (c = new td("onChange", "change", null, c, d), a.push({ event: c, listeners: b }));
}
var pe = null, qe = null;
function re(a) {
  se(a, 0);
}
function te(a) {
  var b = ue(a);
  if (Wa(b))
    return a;
}
function ve(a, b) {
  if ("change" === a)
    return b;
}
var we = false;
if (ia) {
  var xe;
  if (ia) {
    var ye = "oninput" in document;
    if (!ye) {
      var ze = document.createElement("div");
      ze.setAttribute("oninput", "return;");
      ye = "function" === typeof ze.oninput;
    }
    xe = ye;
  } else
    xe = false;
  we = xe && (!document.documentMode || 9 < document.documentMode);
}
function Ae() {
  pe && (pe.detachEvent("onpropertychange", Be), qe = pe = null);
}
function Be(a) {
  if ("value" === a.propertyName && te(qe)) {
    var b = [];
    ne(b, qe, a, xb(a));
    Jb(re, b);
  }
}
function Ce(a, b, c) {
  "focusin" === a ? (Ae(), pe = b, qe = c, pe.attachEvent("onpropertychange", Be)) : "focusout" === a && Ae();
}
function De(a) {
  if ("selectionchange" === a || "keyup" === a || "keydown" === a)
    return te(qe);
}
function Ee(a, b) {
  if ("click" === a)
    return te(b);
}
function Fe(a, b) {
  if ("input" === a || "change" === a)
    return te(b);
}
function Ge(a, b) {
  return a === b && (0 !== a || 1 / a === 1 / b) || a !== a && b !== b;
}
var He = "function" === typeof Object.is ? Object.is : Ge;
function Ie(a, b) {
  if (He(a, b))
    return true;
  if ("object" !== typeof a || null === a || "object" !== typeof b || null === b)
    return false;
  var c = Object.keys(a), d = Object.keys(b);
  if (c.length !== d.length)
    return false;
  for (d = 0; d < c.length; d++) {
    var e = c[d];
    if (!ja.call(b, e) || !He(a[e], b[e]))
      return false;
  }
  return true;
}
function Je(a) {
  for (; a && a.firstChild; )
    a = a.firstChild;
  return a;
}
function Ke(a, b) {
  var c = Je(a);
  a = 0;
  for (var d; c; ) {
    if (3 === c.nodeType) {
      d = a + c.textContent.length;
      if (a <= b && d >= b)
        return { node: c, offset: b - a };
      a = d;
    }
    a: {
      for (; c; ) {
        if (c.nextSibling) {
          c = c.nextSibling;
          break a;
        }
        c = c.parentNode;
      }
      c = void 0;
    }
    c = Je(c);
  }
}
function Le(a, b) {
  return a && b ? a === b ? true : a && 3 === a.nodeType ? false : b && 3 === b.nodeType ? Le(a, b.parentNode) : "contains" in a ? a.contains(b) : a.compareDocumentPosition ? !!(a.compareDocumentPosition(b) & 16) : false : false;
}
function Me() {
  for (var a = window, b = Xa(); b instanceof a.HTMLIFrameElement; ) {
    try {
      var c = "string" === typeof b.contentWindow.location.href;
    } catch (d) {
      c = false;
    }
    if (c)
      a = b.contentWindow;
    else
      break;
    b = Xa(a.document);
  }
  return b;
}
function Ne(a) {
  var b = a && a.nodeName && a.nodeName.toLowerCase();
  return b && ("input" === b && ("text" === a.type || "search" === a.type || "tel" === a.type || "url" === a.type || "password" === a.type) || "textarea" === b || "true" === a.contentEditable);
}
function Oe(a) {
  var b = Me(), c = a.focusedElem, d = a.selectionRange;
  if (b !== c && c && c.ownerDocument && Le(c.ownerDocument.documentElement, c)) {
    if (null !== d && Ne(c)) {
      if (b = d.start, a = d.end, void 0 === a && (a = b), "selectionStart" in c)
        c.selectionStart = b, c.selectionEnd = Math.min(a, c.value.length);
      else if (a = (b = c.ownerDocument || document) && b.defaultView || window, a.getSelection) {
        a = a.getSelection();
        var e = c.textContent.length, f2 = Math.min(d.start, e);
        d = void 0 === d.end ? f2 : Math.min(d.end, e);
        !a.extend && f2 > d && (e = d, d = f2, f2 = e);
        e = Ke(c, f2);
        var g = Ke(
          c,
          d
        );
        e && g && (1 !== a.rangeCount || a.anchorNode !== e.node || a.anchorOffset !== e.offset || a.focusNode !== g.node || a.focusOffset !== g.offset) && (b = b.createRange(), b.setStart(e.node, e.offset), a.removeAllRanges(), f2 > d ? (a.addRange(b), a.extend(g.node, g.offset)) : (b.setEnd(g.node, g.offset), a.addRange(b)));
      }
    }
    b = [];
    for (a = c; a = a.parentNode; )
      1 === a.nodeType && b.push({ element: a, left: a.scrollLeft, top: a.scrollTop });
    "function" === typeof c.focus && c.focus();
    for (c = 0; c < b.length; c++)
      a = b[c], a.element.scrollLeft = a.left, a.element.scrollTop = a.top;
  }
}
var Pe = ia && "documentMode" in document && 11 >= document.documentMode, Qe = null, Re = null, Se = null, Te = false;
function Ue(a, b, c) {
  var d = c.window === c ? c.document : 9 === c.nodeType ? c : c.ownerDocument;
  Te || null == Qe || Qe !== Xa(d) || (d = Qe, "selectionStart" in d && Ne(d) ? d = { start: d.selectionStart, end: d.selectionEnd } : (d = (d.ownerDocument && d.ownerDocument.defaultView || window).getSelection(), d = { anchorNode: d.anchorNode, anchorOffset: d.anchorOffset, focusNode: d.focusNode, focusOffset: d.focusOffset }), Se && Ie(Se, d) || (Se = d, d = oe(Re, "onSelect"), 0 < d.length && (b = new td("onSelect", "select", null, b, c), a.push({ event: b, listeners: d }), b.target = Qe)));
}
function Ve(a, b) {
  var c = {};
  c[a.toLowerCase()] = b.toLowerCase();
  c["Webkit" + a] = "webkit" + b;
  c["Moz" + a] = "moz" + b;
  return c;
}
var We = { animationend: Ve("Animation", "AnimationEnd"), animationiteration: Ve("Animation", "AnimationIteration"), animationstart: Ve("Animation", "AnimationStart"), transitionend: Ve("Transition", "TransitionEnd") }, Xe = {}, Ye = {};
ia && (Ye = document.createElement("div").style, "AnimationEvent" in window || (delete We.animationend.animation, delete We.animationiteration.animation, delete We.animationstart.animation), "TransitionEvent" in window || delete We.transitionend.transition);
function Ze(a) {
  if (Xe[a])
    return Xe[a];
  if (!We[a])
    return a;
  var b = We[a], c;
  for (c in b)
    if (b.hasOwnProperty(c) && c in Ye)
      return Xe[a] = b[c];
  return a;
}
var $e = Ze("animationend"), af = Ze("animationiteration"), bf = Ze("animationstart"), cf = Ze("transitionend"), df = /* @__PURE__ */ new Map(), ef = "abort auxClick cancel canPlay canPlayThrough click close contextMenu copy cut drag dragEnd dragEnter dragExit dragLeave dragOver dragStart drop durationChange emptied encrypted ended error gotPointerCapture input invalid keyDown keyPress keyUp load loadedData loadedMetadata loadStart lostPointerCapture mouseDown mouseMove mouseOut mouseOver mouseUp paste pause play playing pointerCancel pointerDown pointerMove pointerOut pointerOver pointerUp progress rateChange reset resize seeked seeking stalled submit suspend timeUpdate touchCancel touchEnd touchStart volumeChange scroll toggle touchMove waiting wheel".split(" ");
function ff(a, b) {
  df.set(a, b);
  fa(b, [a]);
}
for (var gf = 0; gf < ef.length; gf++) {
  var hf = ef[gf], jf = hf.toLowerCase(), kf = hf[0].toUpperCase() + hf.slice(1);
  ff(jf, "on" + kf);
}
ff($e, "onAnimationEnd");
ff(af, "onAnimationIteration");
ff(bf, "onAnimationStart");
ff("dblclick", "onDoubleClick");
ff("focusin", "onFocus");
ff("focusout", "onBlur");
ff(cf, "onTransitionEnd");
ha("onMouseEnter", ["mouseout", "mouseover"]);
ha("onMouseLeave", ["mouseout", "mouseover"]);
ha("onPointerEnter", ["pointerout", "pointerover"]);
ha("onPointerLeave", ["pointerout", "pointerover"]);
fa("onChange", "change click focusin focusout input keydown keyup selectionchange".split(" "));
fa("onSelect", "focusout contextmenu dragend focusin keydown keyup mousedown mouseup selectionchange".split(" "));
fa("onBeforeInput", ["compositionend", "keypress", "textInput", "paste"]);
fa("onCompositionEnd", "compositionend focusout keydown keypress keyup mousedown".split(" "));
fa("onCompositionStart", "compositionstart focusout keydown keypress keyup mousedown".split(" "));
fa("onCompositionUpdate", "compositionupdate focusout keydown keypress keyup mousedown".split(" "));
var lf = "abort canplay canplaythrough durationchange emptied encrypted ended error loadeddata loadedmetadata loadstart pause play playing progress ratechange resize seeked seeking stalled suspend timeupdate volumechange waiting".split(" "), mf = new Set("cancel close invalid load scroll toggle".split(" ").concat(lf));
function nf(a, b, c) {
  var d = a.type || "unknown-event";
  a.currentTarget = c;
  Ub(d, b, void 0, a);
  a.currentTarget = null;
}
function se(a, b) {
  b = 0 !== (b & 4);
  for (var c = 0; c < a.length; c++) {
    var d = a[c], e = d.event;
    d = d.listeners;
    a: {
      var f2 = void 0;
      if (b)
        for (var g = d.length - 1; 0 <= g; g--) {
          var h = d[g], k2 = h.instance, l2 = h.currentTarget;
          h = h.listener;
          if (k2 !== f2 && e.isPropagationStopped())
            break a;
          nf(e, h, l2);
          f2 = k2;
        }
      else
        for (g = 0; g < d.length; g++) {
          h = d[g];
          k2 = h.instance;
          l2 = h.currentTarget;
          h = h.listener;
          if (k2 !== f2 && e.isPropagationStopped())
            break a;
          nf(e, h, l2);
          f2 = k2;
        }
    }
  }
  if (Qb)
    throw a = Rb, Qb = false, Rb = null, a;
}
function D(a, b) {
  var c = b[of];
  void 0 === c && (c = b[of] = /* @__PURE__ */ new Set());
  var d = a + "__bubble";
  c.has(d) || (pf(b, a, 2, false), c.add(d));
}
function qf(a, b, c) {
  var d = 0;
  b && (d |= 4);
  pf(c, a, d, b);
}
var rf = "_reactListening" + Math.random().toString(36).slice(2);
function sf(a) {
  if (!a[rf]) {
    a[rf] = true;
    da.forEach(function(b2) {
      "selectionchange" !== b2 && (mf.has(b2) || qf(b2, false, a), qf(b2, true, a));
    });
    var b = 9 === a.nodeType ? a : a.ownerDocument;
    null === b || b[rf] || (b[rf] = true, qf("selectionchange", false, b));
  }
}
function pf(a, b, c, d) {
  switch (jd(b)) {
    case 1:
      var e = ed;
      break;
    case 4:
      e = gd;
      break;
    default:
      e = fd;
  }
  c = e.bind(null, b, c, a);
  e = void 0;
  !Lb || "touchstart" !== b && "touchmove" !== b && "wheel" !== b || (e = true);
  d ? void 0 !== e ? a.addEventListener(b, c, { capture: true, passive: e }) : a.addEventListener(b, c, true) : void 0 !== e ? a.addEventListener(b, c, { passive: e }) : a.addEventListener(b, c, false);
}
function hd(a, b, c, d, e) {
  var f2 = d;
  if (0 === (b & 1) && 0 === (b & 2) && null !== d)
    a:
      for (; ; ) {
        if (null === d)
          return;
        var g = d.tag;
        if (3 === g || 4 === g) {
          var h = d.stateNode.containerInfo;
          if (h === e || 8 === h.nodeType && h.parentNode === e)
            break;
          if (4 === g)
            for (g = d.return; null !== g; ) {
              var k2 = g.tag;
              if (3 === k2 || 4 === k2) {
                if (k2 = g.stateNode.containerInfo, k2 === e || 8 === k2.nodeType && k2.parentNode === e)
                  return;
              }
              g = g.return;
            }
          for (; null !== h; ) {
            g = Wc(h);
            if (null === g)
              return;
            k2 = g.tag;
            if (5 === k2 || 6 === k2) {
              d = f2 = g;
              continue a;
            }
            h = h.parentNode;
          }
        }
        d = d.return;
      }
  Jb(function() {
    var d2 = f2, e2 = xb(c), g2 = [];
    a: {
      var h2 = df.get(a);
      if (void 0 !== h2) {
        var k3 = td, n2 = a;
        switch (a) {
          case "keypress":
            if (0 === od(c))
              break a;
          case "keydown":
          case "keyup":
            k3 = Rd;
            break;
          case "focusin":
            n2 = "focus";
            k3 = Fd;
            break;
          case "focusout":
            n2 = "blur";
            k3 = Fd;
            break;
          case "beforeblur":
          case "afterblur":
            k3 = Fd;
            break;
          case "click":
            if (2 === c.button)
              break a;
          case "auxclick":
          case "dblclick":
          case "mousedown":
          case "mousemove":
          case "mouseup":
          case "mouseout":
          case "mouseover":
          case "contextmenu":
            k3 = Bd;
            break;
          case "drag":
          case "dragend":
          case "dragenter":
          case "dragexit":
          case "dragleave":
          case "dragover":
          case "dragstart":
          case "drop":
            k3 = Dd;
            break;
          case "touchcancel":
          case "touchend":
          case "touchmove":
          case "touchstart":
            k3 = Vd;
            break;
          case $e:
          case af:
          case bf:
            k3 = Hd;
            break;
          case cf:
            k3 = Xd;
            break;
          case "scroll":
            k3 = vd;
            break;
          case "wheel":
            k3 = Zd;
            break;
          case "copy":
          case "cut":
          case "paste":
            k3 = Jd;
            break;
          case "gotpointercapture":
          case "lostpointercapture":
          case "pointercancel":
          case "pointerdown":
          case "pointermove":
          case "pointerout":
          case "pointerover":
          case "pointerup":
            k3 = Td;
        }
        var t2 = 0 !== (b & 4), J2 = !t2 && "scroll" === a, x2 = t2 ? null !== h2 ? h2 + "Capture" : null : h2;
        t2 = [];
        for (var w2 = d2, u2; null !== w2; ) {
          u2 = w2;
          var F2 = u2.stateNode;
          5 === u2.tag && null !== F2 && (u2 = F2, null !== x2 && (F2 = Kb(w2, x2), null != F2 && t2.push(tf(w2, F2, u2))));
          if (J2)
            break;
          w2 = w2.return;
        }
        0 < t2.length && (h2 = new k3(h2, n2, null, c, e2), g2.push({ event: h2, listeners: t2 }));
      }
    }
    if (0 === (b & 7)) {
      a: {
        h2 = "mouseover" === a || "pointerover" === a;
        k3 = "mouseout" === a || "pointerout" === a;
        if (h2 && c !== wb && (n2 = c.relatedTarget || c.fromElement) && (Wc(n2) || n2[uf]))
          break a;
        if (k3 || h2) {
          h2 = e2.window === e2 ? e2 : (h2 = e2.ownerDocument) ? h2.defaultView || h2.parentWindow : window;
          if (k3) {
            if (n2 = c.relatedTarget || c.toElement, k3 = d2, n2 = n2 ? Wc(n2) : null, null !== n2 && (J2 = Vb(n2), n2 !== J2 || 5 !== n2.tag && 6 !== n2.tag))
              n2 = null;
          } else
            k3 = null, n2 = d2;
          if (k3 !== n2) {
            t2 = Bd;
            F2 = "onMouseLeave";
            x2 = "onMouseEnter";
            w2 = "mouse";
            if ("pointerout" === a || "pointerover" === a)
              t2 = Td, F2 = "onPointerLeave", x2 = "onPointerEnter", w2 = "pointer";
            J2 = null == k3 ? h2 : ue(k3);
            u2 = null == n2 ? h2 : ue(n2);
            h2 = new t2(F2, w2 + "leave", k3, c, e2);
            h2.target = J2;
            h2.relatedTarget = u2;
            F2 = null;
            Wc(e2) === d2 && (t2 = new t2(x2, w2 + "enter", n2, c, e2), t2.target = u2, t2.relatedTarget = J2, F2 = t2);
            J2 = F2;
            if (k3 && n2)
              b: {
                t2 = k3;
                x2 = n2;
                w2 = 0;
                for (u2 = t2; u2; u2 = vf(u2))
                  w2++;
                u2 = 0;
                for (F2 = x2; F2; F2 = vf(F2))
                  u2++;
                for (; 0 < w2 - u2; )
                  t2 = vf(t2), w2--;
                for (; 0 < u2 - w2; )
                  x2 = vf(x2), u2--;
                for (; w2--; ) {
                  if (t2 === x2 || null !== x2 && t2 === x2.alternate)
                    break b;
                  t2 = vf(t2);
                  x2 = vf(x2);
                }
                t2 = null;
              }
            else
              t2 = null;
            null !== k3 && wf(g2, h2, k3, t2, false);
            null !== n2 && null !== J2 && wf(g2, J2, n2, t2, true);
          }
        }
      }
      a: {
        h2 = d2 ? ue(d2) : window;
        k3 = h2.nodeName && h2.nodeName.toLowerCase();
        if ("select" === k3 || "input" === k3 && "file" === h2.type)
          var na = ve;
        else if (me(h2))
          if (we)
            na = Fe;
          else {
            na = De;
            var xa = Ce;
          }
        else
          (k3 = h2.nodeName) && "input" === k3.toLowerCase() && ("checkbox" === h2.type || "radio" === h2.type) && (na = Ee);
        if (na && (na = na(a, d2))) {
          ne(g2, na, c, e2);
          break a;
        }
        xa && xa(a, h2, d2);
        "focusout" === a && (xa = h2._wrapperState) && xa.controlled && "number" === h2.type && cb(h2, "number", h2.value);
      }
      xa = d2 ? ue(d2) : window;
      switch (a) {
        case "focusin":
          if (me(xa) || "true" === xa.contentEditable)
            Qe = xa, Re = d2, Se = null;
          break;
        case "focusout":
          Se = Re = Qe = null;
          break;
        case "mousedown":
          Te = true;
          break;
        case "contextmenu":
        case "mouseup":
        case "dragend":
          Te = false;
          Ue(g2, c, e2);
          break;
        case "selectionchange":
          if (Pe)
            break;
        case "keydown":
        case "keyup":
          Ue(g2, c, e2);
      }
      var $a;
      if (ae)
        b: {
          switch (a) {
            case "compositionstart":
              var ba = "onCompositionStart";
              break b;
            case "compositionend":
              ba = "onCompositionEnd";
              break b;
            case "compositionupdate":
              ba = "onCompositionUpdate";
              break b;
          }
          ba = void 0;
        }
      else
        ie ? ge(a, c) && (ba = "onCompositionEnd") : "keydown" === a && 229 === c.keyCode && (ba = "onCompositionStart");
      ba && (de && "ko" !== c.locale && (ie || "onCompositionStart" !== ba ? "onCompositionEnd" === ba && ie && ($a = nd()) : (kd = e2, ld = "value" in kd ? kd.value : kd.textContent, ie = true)), xa = oe(d2, ba), 0 < xa.length && (ba = new Ld(ba, a, null, c, e2), g2.push({ event: ba, listeners: xa }), $a ? ba.data = $a : ($a = he(c), null !== $a && (ba.data = $a))));
      if ($a = ce ? je(a, c) : ke(a, c))
        d2 = oe(d2, "onBeforeInput"), 0 < d2.length && (e2 = new Ld("onBeforeInput", "beforeinput", null, c, e2), g2.push({ event: e2, listeners: d2 }), e2.data = $a);
    }
    se(g2, b);
  });
}
function tf(a, b, c) {
  return { instance: a, listener: b, currentTarget: c };
}
function oe(a, b) {
  for (var c = b + "Capture", d = []; null !== a; ) {
    var e = a, f2 = e.stateNode;
    5 === e.tag && null !== f2 && (e = f2, f2 = Kb(a, c), null != f2 && d.unshift(tf(a, f2, e)), f2 = Kb(a, b), null != f2 && d.push(tf(a, f2, e)));
    a = a.return;
  }
  return d;
}
function vf(a) {
  if (null === a)
    return null;
  do
    a = a.return;
  while (a && 5 !== a.tag);
  return a ? a : null;
}
function wf(a, b, c, d, e) {
  for (var f2 = b._reactName, g = []; null !== c && c !== d; ) {
    var h = c, k2 = h.alternate, l2 = h.stateNode;
    if (null !== k2 && k2 === d)
      break;
    5 === h.tag && null !== l2 && (h = l2, e ? (k2 = Kb(c, f2), null != k2 && g.unshift(tf(c, k2, h))) : e || (k2 = Kb(c, f2), null != k2 && g.push(tf(c, k2, h))));
    c = c.return;
  }
  0 !== g.length && a.push({ event: b, listeners: g });
}
var xf = /\r\n?/g, yf = /\u0000|\uFFFD/g;
function zf(a) {
  return ("string" === typeof a ? a : "" + a).replace(xf, "\n").replace(yf, "");
}
function Af(a, b, c) {
  b = zf(b);
  if (zf(a) !== b && c)
    throw Error(p(425));
}
function Bf() {
}
var Cf = null, Df = null;
function Ef(a, b) {
  return "textarea" === a || "noscript" === a || "string" === typeof b.children || "number" === typeof b.children || "object" === typeof b.dangerouslySetInnerHTML && null !== b.dangerouslySetInnerHTML && null != b.dangerouslySetInnerHTML.__html;
}
var Ff = "function" === typeof setTimeout ? setTimeout : void 0, Gf = "function" === typeof clearTimeout ? clearTimeout : void 0, Hf = "function" === typeof Promise ? Promise : void 0, Jf = "function" === typeof queueMicrotask ? queueMicrotask : "undefined" !== typeof Hf ? function(a) {
  return Hf.resolve(null).then(a).catch(If);
} : Ff;
function If(a) {
  setTimeout(function() {
    throw a;
  });
}
function Kf(a, b) {
  var c = b, d = 0;
  do {
    var e = c.nextSibling;
    a.removeChild(c);
    if (e && 8 === e.nodeType)
      if (c = e.data, "/$" === c) {
        if (0 === d) {
          a.removeChild(e);
          bd(b);
          return;
        }
        d--;
      } else
        "$" !== c && "$?" !== c && "$!" !== c || d++;
    c = e;
  } while (c);
  bd(b);
}
function Lf(a) {
  for (; null != a; a = a.nextSibling) {
    var b = a.nodeType;
    if (1 === b || 3 === b)
      break;
    if (8 === b) {
      b = a.data;
      if ("$" === b || "$!" === b || "$?" === b)
        break;
      if ("/$" === b)
        return null;
    }
  }
  return a;
}
function Mf(a) {
  a = a.previousSibling;
  for (var b = 0; a; ) {
    if (8 === a.nodeType) {
      var c = a.data;
      if ("$" === c || "$!" === c || "$?" === c) {
        if (0 === b)
          return a;
        b--;
      } else
        "/$" === c && b++;
    }
    a = a.previousSibling;
  }
  return null;
}
var Nf = Math.random().toString(36).slice(2), Of = "__reactFiber$" + Nf, Pf = "__reactProps$" + Nf, uf = "__reactContainer$" + Nf, of = "__reactEvents$" + Nf, Qf = "__reactListeners$" + Nf, Rf = "__reactHandles$" + Nf;
function Wc(a) {
  var b = a[Of];
  if (b)
    return b;
  for (var c = a.parentNode; c; ) {
    if (b = c[uf] || c[Of]) {
      c = b.alternate;
      if (null !== b.child || null !== c && null !== c.child)
        for (a = Mf(a); null !== a; ) {
          if (c = a[Of])
            return c;
          a = Mf(a);
        }
      return b;
    }
    a = c;
    c = a.parentNode;
  }
  return null;
}
function Cb(a) {
  a = a[Of] || a[uf];
  return !a || 5 !== a.tag && 6 !== a.tag && 13 !== a.tag && 3 !== a.tag ? null : a;
}
function ue(a) {
  if (5 === a.tag || 6 === a.tag)
    return a.stateNode;
  throw Error(p(33));
}
function Db(a) {
  return a[Pf] || null;
}
var Sf = [], Tf = -1;
function Uf(a) {
  return { current: a };
}
function E(a) {
  0 > Tf || (a.current = Sf[Tf], Sf[Tf] = null, Tf--);
}
function G(a, b) {
  Tf++;
  Sf[Tf] = a.current;
  a.current = b;
}
var Vf = {}, H = Uf(Vf), Wf = Uf(false), Xf = Vf;
function Yf(a, b) {
  var c = a.type.contextTypes;
  if (!c)
    return Vf;
  var d = a.stateNode;
  if (d && d.__reactInternalMemoizedUnmaskedChildContext === b)
    return d.__reactInternalMemoizedMaskedChildContext;
  var e = {}, f2;
  for (f2 in c)
    e[f2] = b[f2];
  d && (a = a.stateNode, a.__reactInternalMemoizedUnmaskedChildContext = b, a.__reactInternalMemoizedMaskedChildContext = e);
  return e;
}
function Zf(a) {
  a = a.childContextTypes;
  return null !== a && void 0 !== a;
}
function $f() {
  E(Wf);
  E(H);
}
function ag(a, b, c) {
  if (H.current !== Vf)
    throw Error(p(168));
  G(H, b);
  G(Wf, c);
}
function bg(a, b, c) {
  var d = a.stateNode;
  b = b.childContextTypes;
  if ("function" !== typeof d.getChildContext)
    return c;
  d = d.getChildContext();
  for (var e in d)
    if (!(e in b))
      throw Error(p(108, Ra(a) || "Unknown", e));
  return A({}, c, d);
}
function cg(a) {
  a = (a = a.stateNode) && a.__reactInternalMemoizedMergedChildContext || Vf;
  Xf = H.current;
  G(H, a);
  G(Wf, Wf.current);
  return true;
}
function dg(a, b, c) {
  var d = a.stateNode;
  if (!d)
    throw Error(p(169));
  c ? (a = bg(a, b, Xf), d.__reactInternalMemoizedMergedChildContext = a, E(Wf), E(H), G(H, a)) : E(Wf);
  G(Wf, c);
}
var eg = null, fg = false, gg = false;
function hg(a) {
  null === eg ? eg = [a] : eg.push(a);
}
function ig(a) {
  fg = true;
  hg(a);
}
function jg() {
  if (!gg && null !== eg) {
    gg = true;
    var a = 0, b = C;
    try {
      var c = eg;
      for (C = 1; a < c.length; a++) {
        var d = c[a];
        do
          d = d(true);
        while (null !== d);
      }
      eg = null;
      fg = false;
    } catch (e) {
      throw null !== eg && (eg = eg.slice(a + 1)), ac(fc, jg), e;
    } finally {
      C = b, gg = false;
    }
  }
  return null;
}
var kg = [], lg = 0, mg = null, ng = 0, og = [], pg = 0, qg = null, rg = 1, sg = "";
function tg(a, b) {
  kg[lg++] = ng;
  kg[lg++] = mg;
  mg = a;
  ng = b;
}
function ug(a, b, c) {
  og[pg++] = rg;
  og[pg++] = sg;
  og[pg++] = qg;
  qg = a;
  var d = rg;
  a = sg;
  var e = 32 - oc(d) - 1;
  d &= ~(1 << e);
  c += 1;
  var f2 = 32 - oc(b) + e;
  if (30 < f2) {
    var g = e - e % 5;
    f2 = (d & (1 << g) - 1).toString(32);
    d >>= g;
    e -= g;
    rg = 1 << 32 - oc(b) + e | c << e | d;
    sg = f2 + a;
  } else
    rg = 1 << f2 | c << e | d, sg = a;
}
function vg(a) {
  null !== a.return && (tg(a, 1), ug(a, 1, 0));
}
function wg(a) {
  for (; a === mg; )
    mg = kg[--lg], kg[lg] = null, ng = kg[--lg], kg[lg] = null;
  for (; a === qg; )
    qg = og[--pg], og[pg] = null, sg = og[--pg], og[pg] = null, rg = og[--pg], og[pg] = null;
}
var xg = null, yg = null, I = false, zg = null;
function Ag(a, b) {
  var c = Bg(5, null, null, 0);
  c.elementType = "DELETED";
  c.stateNode = b;
  c.return = a;
  b = a.deletions;
  null === b ? (a.deletions = [c], a.flags |= 16) : b.push(c);
}
function Cg(a, b) {
  switch (a.tag) {
    case 5:
      var c = a.type;
      b = 1 !== b.nodeType || c.toLowerCase() !== b.nodeName.toLowerCase() ? null : b;
      return null !== b ? (a.stateNode = b, xg = a, yg = Lf(b.firstChild), true) : false;
    case 6:
      return b = "" === a.pendingProps || 3 !== b.nodeType ? null : b, null !== b ? (a.stateNode = b, xg = a, yg = null, true) : false;
    case 13:
      return b = 8 !== b.nodeType ? null : b, null !== b ? (c = null !== qg ? { id: rg, overflow: sg } : null, a.memoizedState = { dehydrated: b, treeContext: c, retryLane: 1073741824 }, c = Bg(18, null, null, 0), c.stateNode = b, c.return = a, a.child = c, xg = a, yg = null, true) : false;
    default:
      return false;
  }
}
function Dg(a) {
  return 0 !== (a.mode & 1) && 0 === (a.flags & 128);
}
function Eg(a) {
  if (I) {
    var b = yg;
    if (b) {
      var c = b;
      if (!Cg(a, b)) {
        if (Dg(a))
          throw Error(p(418));
        b = Lf(c.nextSibling);
        var d = xg;
        b && Cg(a, b) ? Ag(d, c) : (a.flags = a.flags & -4097 | 2, I = false, xg = a);
      }
    } else {
      if (Dg(a))
        throw Error(p(418));
      a.flags = a.flags & -4097 | 2;
      I = false;
      xg = a;
    }
  }
}
function Fg(a) {
  for (a = a.return; null !== a && 5 !== a.tag && 3 !== a.tag && 13 !== a.tag; )
    a = a.return;
  xg = a;
}
function Gg(a) {
  if (a !== xg)
    return false;
  if (!I)
    return Fg(a), I = true, false;
  var b;
  (b = 3 !== a.tag) && !(b = 5 !== a.tag) && (b = a.type, b = "head" !== b && "body" !== b && !Ef(a.type, a.memoizedProps));
  if (b && (b = yg)) {
    if (Dg(a))
      throw Hg(), Error(p(418));
    for (; b; )
      Ag(a, b), b = Lf(b.nextSibling);
  }
  Fg(a);
  if (13 === a.tag) {
    a = a.memoizedState;
    a = null !== a ? a.dehydrated : null;
    if (!a)
      throw Error(p(317));
    a: {
      a = a.nextSibling;
      for (b = 0; a; ) {
        if (8 === a.nodeType) {
          var c = a.data;
          if ("/$" === c) {
            if (0 === b) {
              yg = Lf(a.nextSibling);
              break a;
            }
            b--;
          } else
            "$" !== c && "$!" !== c && "$?" !== c || b++;
        }
        a = a.nextSibling;
      }
      yg = null;
    }
  } else
    yg = xg ? Lf(a.stateNode.nextSibling) : null;
  return true;
}
function Hg() {
  for (var a = yg; a; )
    a = Lf(a.nextSibling);
}
function Ig() {
  yg = xg = null;
  I = false;
}
function Jg(a) {
  null === zg ? zg = [a] : zg.push(a);
}
var Kg = ua.ReactCurrentBatchConfig;
function Lg(a, b, c) {
  a = c.ref;
  if (null !== a && "function" !== typeof a && "object" !== typeof a) {
    if (c._owner) {
      c = c._owner;
      if (c) {
        if (1 !== c.tag)
          throw Error(p(309));
        var d = c.stateNode;
      }
      if (!d)
        throw Error(p(147, a));
      var e = d, f2 = "" + a;
      if (null !== b && null !== b.ref && "function" === typeof b.ref && b.ref._stringRef === f2)
        return b.ref;
      b = function(a2) {
        var b2 = e.refs;
        null === a2 ? delete b2[f2] : b2[f2] = a2;
      };
      b._stringRef = f2;
      return b;
    }
    if ("string" !== typeof a)
      throw Error(p(284));
    if (!c._owner)
      throw Error(p(290, a));
  }
  return a;
}
function Mg(a, b) {
  a = Object.prototype.toString.call(b);
  throw Error(p(31, "[object Object]" === a ? "object with keys {" + Object.keys(b).join(", ") + "}" : a));
}
function Ng(a) {
  var b = a._init;
  return b(a._payload);
}
function Og(a) {
  function b(b2, c2) {
    if (a) {
      var d2 = b2.deletions;
      null === d2 ? (b2.deletions = [c2], b2.flags |= 16) : d2.push(c2);
    }
  }
  function c(c2, d2) {
    if (!a)
      return null;
    for (; null !== d2; )
      b(c2, d2), d2 = d2.sibling;
    return null;
  }
  function d(a2, b2) {
    for (a2 = /* @__PURE__ */ new Map(); null !== b2; )
      null !== b2.key ? a2.set(b2.key, b2) : a2.set(b2.index, b2), b2 = b2.sibling;
    return a2;
  }
  function e(a2, b2) {
    a2 = Pg(a2, b2);
    a2.index = 0;
    a2.sibling = null;
    return a2;
  }
  function f2(b2, c2, d2) {
    b2.index = d2;
    if (!a)
      return b2.flags |= 1048576, c2;
    d2 = b2.alternate;
    if (null !== d2)
      return d2 = d2.index, d2 < c2 ? (b2.flags |= 2, c2) : d2;
    b2.flags |= 2;
    return c2;
  }
  function g(b2) {
    a && null === b2.alternate && (b2.flags |= 2);
    return b2;
  }
  function h(a2, b2, c2, d2) {
    if (null === b2 || 6 !== b2.tag)
      return b2 = Qg(c2, a2.mode, d2), b2.return = a2, b2;
    b2 = e(b2, c2);
    b2.return = a2;
    return b2;
  }
  function k2(a2, b2, c2, d2) {
    var f3 = c2.type;
    if (f3 === ya)
      return m2(a2, b2, c2.props.children, d2, c2.key);
    if (null !== b2 && (b2.elementType === f3 || "object" === typeof f3 && null !== f3 && f3.$$typeof === Ha && Ng(f3) === b2.type))
      return d2 = e(b2, c2.props), d2.ref = Lg(a2, b2, c2), d2.return = a2, d2;
    d2 = Rg(c2.type, c2.key, c2.props, null, a2.mode, d2);
    d2.ref = Lg(a2, b2, c2);
    d2.return = a2;
    return d2;
  }
  function l2(a2, b2, c2, d2) {
    if (null === b2 || 4 !== b2.tag || b2.stateNode.containerInfo !== c2.containerInfo || b2.stateNode.implementation !== c2.implementation)
      return b2 = Sg(c2, a2.mode, d2), b2.return = a2, b2;
    b2 = e(b2, c2.children || []);
    b2.return = a2;
    return b2;
  }
  function m2(a2, b2, c2, d2, f3) {
    if (null === b2 || 7 !== b2.tag)
      return b2 = Tg(c2, a2.mode, d2, f3), b2.return = a2, b2;
    b2 = e(b2, c2);
    b2.return = a2;
    return b2;
  }
  function q2(a2, b2, c2) {
    if ("string" === typeof b2 && "" !== b2 || "number" === typeof b2)
      return b2 = Qg("" + b2, a2.mode, c2), b2.return = a2, b2;
    if ("object" === typeof b2 && null !== b2) {
      switch (b2.$$typeof) {
        case va:
          return c2 = Rg(b2.type, b2.key, b2.props, null, a2.mode, c2), c2.ref = Lg(a2, null, b2), c2.return = a2, c2;
        case wa:
          return b2 = Sg(b2, a2.mode, c2), b2.return = a2, b2;
        case Ha:
          var d2 = b2._init;
          return q2(a2, d2(b2._payload), c2);
      }
      if (eb(b2) || Ka(b2))
        return b2 = Tg(b2, a2.mode, c2, null), b2.return = a2, b2;
      Mg(a2, b2);
    }
    return null;
  }
  function r2(a2, b2, c2, d2) {
    var e2 = null !== b2 ? b2.key : null;
    if ("string" === typeof c2 && "" !== c2 || "number" === typeof c2)
      return null !== e2 ? null : h(a2, b2, "" + c2, d2);
    if ("object" === typeof c2 && null !== c2) {
      switch (c2.$$typeof) {
        case va:
          return c2.key === e2 ? k2(a2, b2, c2, d2) : null;
        case wa:
          return c2.key === e2 ? l2(a2, b2, c2, d2) : null;
        case Ha:
          return e2 = c2._init, r2(
            a2,
            b2,
            e2(c2._payload),
            d2
          );
      }
      if (eb(c2) || Ka(c2))
        return null !== e2 ? null : m2(a2, b2, c2, d2, null);
      Mg(a2, c2);
    }
    return null;
  }
  function y2(a2, b2, c2, d2, e2) {
    if ("string" === typeof d2 && "" !== d2 || "number" === typeof d2)
      return a2 = a2.get(c2) || null, h(b2, a2, "" + d2, e2);
    if ("object" === typeof d2 && null !== d2) {
      switch (d2.$$typeof) {
        case va:
          return a2 = a2.get(null === d2.key ? c2 : d2.key) || null, k2(b2, a2, d2, e2);
        case wa:
          return a2 = a2.get(null === d2.key ? c2 : d2.key) || null, l2(b2, a2, d2, e2);
        case Ha:
          var f3 = d2._init;
          return y2(a2, b2, c2, f3(d2._payload), e2);
      }
      if (eb(d2) || Ka(d2))
        return a2 = a2.get(c2) || null, m2(b2, a2, d2, e2, null);
      Mg(b2, d2);
    }
    return null;
  }
  function n2(e2, g2, h2, k3) {
    for (var l3 = null, m3 = null, u2 = g2, w2 = g2 = 0, x2 = null; null !== u2 && w2 < h2.length; w2++) {
      u2.index > w2 ? (x2 = u2, u2 = null) : x2 = u2.sibling;
      var n3 = r2(e2, u2, h2[w2], k3);
      if (null === n3) {
        null === u2 && (u2 = x2);
        break;
      }
      a && u2 && null === n3.alternate && b(e2, u2);
      g2 = f2(n3, g2, w2);
      null === m3 ? l3 = n3 : m3.sibling = n3;
      m3 = n3;
      u2 = x2;
    }
    if (w2 === h2.length)
      return c(e2, u2), I && tg(e2, w2), l3;
    if (null === u2) {
      for (; w2 < h2.length; w2++)
        u2 = q2(e2, h2[w2], k3), null !== u2 && (g2 = f2(u2, g2, w2), null === m3 ? l3 = u2 : m3.sibling = u2, m3 = u2);
      I && tg(e2, w2);
      return l3;
    }
    for (u2 = d(e2, u2); w2 < h2.length; w2++)
      x2 = y2(u2, e2, w2, h2[w2], k3), null !== x2 && (a && null !== x2.alternate && u2.delete(null === x2.key ? w2 : x2.key), g2 = f2(x2, g2, w2), null === m3 ? l3 = x2 : m3.sibling = x2, m3 = x2);
    a && u2.forEach(function(a2) {
      return b(e2, a2);
    });
    I && tg(e2, w2);
    return l3;
  }
  function t2(e2, g2, h2, k3) {
    var l3 = Ka(h2);
    if ("function" !== typeof l3)
      throw Error(p(150));
    h2 = l3.call(h2);
    if (null == h2)
      throw Error(p(151));
    for (var u2 = l3 = null, m3 = g2, w2 = g2 = 0, x2 = null, n3 = h2.next(); null !== m3 && !n3.done; w2++, n3 = h2.next()) {
      m3.index > w2 ? (x2 = m3, m3 = null) : x2 = m3.sibling;
      var t3 = r2(e2, m3, n3.value, k3);
      if (null === t3) {
        null === m3 && (m3 = x2);
        break;
      }
      a && m3 && null === t3.alternate && b(e2, m3);
      g2 = f2(t3, g2, w2);
      null === u2 ? l3 = t3 : u2.sibling = t3;
      u2 = t3;
      m3 = x2;
    }
    if (n3.done)
      return c(
        e2,
        m3
      ), I && tg(e2, w2), l3;
    if (null === m3) {
      for (; !n3.done; w2++, n3 = h2.next())
        n3 = q2(e2, n3.value, k3), null !== n3 && (g2 = f2(n3, g2, w2), null === u2 ? l3 = n3 : u2.sibling = n3, u2 = n3);
      I && tg(e2, w2);
      return l3;
    }
    for (m3 = d(e2, m3); !n3.done; w2++, n3 = h2.next())
      n3 = y2(m3, e2, w2, n3.value, k3), null !== n3 && (a && null !== n3.alternate && m3.delete(null === n3.key ? w2 : n3.key), g2 = f2(n3, g2, w2), null === u2 ? l3 = n3 : u2.sibling = n3, u2 = n3);
    a && m3.forEach(function(a2) {
      return b(e2, a2);
    });
    I && tg(e2, w2);
    return l3;
  }
  function J2(a2, d2, f3, h2) {
    "object" === typeof f3 && null !== f3 && f3.type === ya && null === f3.key && (f3 = f3.props.children);
    if ("object" === typeof f3 && null !== f3) {
      switch (f3.$$typeof) {
        case va:
          a: {
            for (var k3 = f3.key, l3 = d2; null !== l3; ) {
              if (l3.key === k3) {
                k3 = f3.type;
                if (k3 === ya) {
                  if (7 === l3.tag) {
                    c(a2, l3.sibling);
                    d2 = e(l3, f3.props.children);
                    d2.return = a2;
                    a2 = d2;
                    break a;
                  }
                } else if (l3.elementType === k3 || "object" === typeof k3 && null !== k3 && k3.$$typeof === Ha && Ng(k3) === l3.type) {
                  c(a2, l3.sibling);
                  d2 = e(l3, f3.props);
                  d2.ref = Lg(a2, l3, f3);
                  d2.return = a2;
                  a2 = d2;
                  break a;
                }
                c(a2, l3);
                break;
              } else
                b(a2, l3);
              l3 = l3.sibling;
            }
            f3.type === ya ? (d2 = Tg(f3.props.children, a2.mode, h2, f3.key), d2.return = a2, a2 = d2) : (h2 = Rg(f3.type, f3.key, f3.props, null, a2.mode, h2), h2.ref = Lg(a2, d2, f3), h2.return = a2, a2 = h2);
          }
          return g(a2);
        case wa:
          a: {
            for (l3 = f3.key; null !== d2; ) {
              if (d2.key === l3)
                if (4 === d2.tag && d2.stateNode.containerInfo === f3.containerInfo && d2.stateNode.implementation === f3.implementation) {
                  c(a2, d2.sibling);
                  d2 = e(d2, f3.children || []);
                  d2.return = a2;
                  a2 = d2;
                  break a;
                } else {
                  c(a2, d2);
                  break;
                }
              else
                b(a2, d2);
              d2 = d2.sibling;
            }
            d2 = Sg(f3, a2.mode, h2);
            d2.return = a2;
            a2 = d2;
          }
          return g(a2);
        case Ha:
          return l3 = f3._init, J2(a2, d2, l3(f3._payload), h2);
      }
      if (eb(f3))
        return n2(a2, d2, f3, h2);
      if (Ka(f3))
        return t2(a2, d2, f3, h2);
      Mg(a2, f3);
    }
    return "string" === typeof f3 && "" !== f3 || "number" === typeof f3 ? (f3 = "" + f3, null !== d2 && 6 === d2.tag ? (c(a2, d2.sibling), d2 = e(d2, f3), d2.return = a2, a2 = d2) : (c(a2, d2), d2 = Qg(f3, a2.mode, h2), d2.return = a2, a2 = d2), g(a2)) : c(a2, d2);
  }
  return J2;
}
var Ug = Og(true), Vg = Og(false), Wg = Uf(null), Xg = null, Yg = null, Zg = null;
function $g() {
  Zg = Yg = Xg = null;
}
function ah(a) {
  var b = Wg.current;
  E(Wg);
  a._currentValue = b;
}
function bh(a, b, c) {
  for (; null !== a; ) {
    var d = a.alternate;
    (a.childLanes & b) !== b ? (a.childLanes |= b, null !== d && (d.childLanes |= b)) : null !== d && (d.childLanes & b) !== b && (d.childLanes |= b);
    if (a === c)
      break;
    a = a.return;
  }
}
function ch(a, b) {
  Xg = a;
  Zg = Yg = null;
  a = a.dependencies;
  null !== a && null !== a.firstContext && (0 !== (a.lanes & b) && (dh = true), a.firstContext = null);
}
function eh(a) {
  var b = a._currentValue;
  if (Zg !== a)
    if (a = { context: a, memoizedValue: b, next: null }, null === Yg) {
      if (null === Xg)
        throw Error(p(308));
      Yg = a;
      Xg.dependencies = { lanes: 0, firstContext: a };
    } else
      Yg = Yg.next = a;
  return b;
}
var fh = null;
function gh(a) {
  null === fh ? fh = [a] : fh.push(a);
}
function hh(a, b, c, d) {
  var e = b.interleaved;
  null === e ? (c.next = c, gh(b)) : (c.next = e.next, e.next = c);
  b.interleaved = c;
  return ih(a, d);
}
function ih(a, b) {
  a.lanes |= b;
  var c = a.alternate;
  null !== c && (c.lanes |= b);
  c = a;
  for (a = a.return; null !== a; )
    a.childLanes |= b, c = a.alternate, null !== c && (c.childLanes |= b), c = a, a = a.return;
  return 3 === c.tag ? c.stateNode : null;
}
var jh = false;
function kh(a) {
  a.updateQueue = { baseState: a.memoizedState, firstBaseUpdate: null, lastBaseUpdate: null, shared: { pending: null, interleaved: null, lanes: 0 }, effects: null };
}
function lh(a, b) {
  a = a.updateQueue;
  b.updateQueue === a && (b.updateQueue = { baseState: a.baseState, firstBaseUpdate: a.firstBaseUpdate, lastBaseUpdate: a.lastBaseUpdate, shared: a.shared, effects: a.effects });
}
function mh(a, b) {
  return { eventTime: a, lane: b, tag: 0, payload: null, callback: null, next: null };
}
function nh(a, b, c) {
  var d = a.updateQueue;
  if (null === d)
    return null;
  d = d.shared;
  if (0 !== (K & 2)) {
    var e = d.pending;
    null === e ? b.next = b : (b.next = e.next, e.next = b);
    d.pending = b;
    return ih(a, c);
  }
  e = d.interleaved;
  null === e ? (b.next = b, gh(d)) : (b.next = e.next, e.next = b);
  d.interleaved = b;
  return ih(a, c);
}
function oh(a, b, c) {
  b = b.updateQueue;
  if (null !== b && (b = b.shared, 0 !== (c & 4194240))) {
    var d = b.lanes;
    d &= a.pendingLanes;
    c |= d;
    b.lanes = c;
    Cc(a, c);
  }
}
function ph(a, b) {
  var c = a.updateQueue, d = a.alternate;
  if (null !== d && (d = d.updateQueue, c === d)) {
    var e = null, f2 = null;
    c = c.firstBaseUpdate;
    if (null !== c) {
      do {
        var g = { eventTime: c.eventTime, lane: c.lane, tag: c.tag, payload: c.payload, callback: c.callback, next: null };
        null === f2 ? e = f2 = g : f2 = f2.next = g;
        c = c.next;
      } while (null !== c);
      null === f2 ? e = f2 = b : f2 = f2.next = b;
    } else
      e = f2 = b;
    c = { baseState: d.baseState, firstBaseUpdate: e, lastBaseUpdate: f2, shared: d.shared, effects: d.effects };
    a.updateQueue = c;
    return;
  }
  a = c.lastBaseUpdate;
  null === a ? c.firstBaseUpdate = b : a.next = b;
  c.lastBaseUpdate = b;
}
function qh(a, b, c, d) {
  var e = a.updateQueue;
  jh = false;
  var f2 = e.firstBaseUpdate, g = e.lastBaseUpdate, h = e.shared.pending;
  if (null !== h) {
    e.shared.pending = null;
    var k2 = h, l2 = k2.next;
    k2.next = null;
    null === g ? f2 = l2 : g.next = l2;
    g = k2;
    var m2 = a.alternate;
    null !== m2 && (m2 = m2.updateQueue, h = m2.lastBaseUpdate, h !== g && (null === h ? m2.firstBaseUpdate = l2 : h.next = l2, m2.lastBaseUpdate = k2));
  }
  if (null !== f2) {
    var q2 = e.baseState;
    g = 0;
    m2 = l2 = k2 = null;
    h = f2;
    do {
      var r2 = h.lane, y2 = h.eventTime;
      if ((d & r2) === r2) {
        null !== m2 && (m2 = m2.next = {
          eventTime: y2,
          lane: 0,
          tag: h.tag,
          payload: h.payload,
          callback: h.callback,
          next: null
        });
        a: {
          var n2 = a, t2 = h;
          r2 = b;
          y2 = c;
          switch (t2.tag) {
            case 1:
              n2 = t2.payload;
              if ("function" === typeof n2) {
                q2 = n2.call(y2, q2, r2);
                break a;
              }
              q2 = n2;
              break a;
            case 3:
              n2.flags = n2.flags & -65537 | 128;
            case 0:
              n2 = t2.payload;
              r2 = "function" === typeof n2 ? n2.call(y2, q2, r2) : n2;
              if (null === r2 || void 0 === r2)
                break a;
              q2 = A({}, q2, r2);
              break a;
            case 2:
              jh = true;
          }
        }
        null !== h.callback && 0 !== h.lane && (a.flags |= 64, r2 = e.effects, null === r2 ? e.effects = [h] : r2.push(h));
      } else
        y2 = { eventTime: y2, lane: r2, tag: h.tag, payload: h.payload, callback: h.callback, next: null }, null === m2 ? (l2 = m2 = y2, k2 = q2) : m2 = m2.next = y2, g |= r2;
      h = h.next;
      if (null === h)
        if (h = e.shared.pending, null === h)
          break;
        else
          r2 = h, h = r2.next, r2.next = null, e.lastBaseUpdate = r2, e.shared.pending = null;
    } while (1);
    null === m2 && (k2 = q2);
    e.baseState = k2;
    e.firstBaseUpdate = l2;
    e.lastBaseUpdate = m2;
    b = e.shared.interleaved;
    if (null !== b) {
      e = b;
      do
        g |= e.lane, e = e.next;
      while (e !== b);
    } else
      null === f2 && (e.shared.lanes = 0);
    rh |= g;
    a.lanes = g;
    a.memoizedState = q2;
  }
}
function sh(a, b, c) {
  a = b.effects;
  b.effects = null;
  if (null !== a)
    for (b = 0; b < a.length; b++) {
      var d = a[b], e = d.callback;
      if (null !== e) {
        d.callback = null;
        d = c;
        if ("function" !== typeof e)
          throw Error(p(191, e));
        e.call(d);
      }
    }
}
var th = {}, uh = Uf(th), vh = Uf(th), wh = Uf(th);
function xh(a) {
  if (a === th)
    throw Error(p(174));
  return a;
}
function yh(a, b) {
  G(wh, b);
  G(vh, a);
  G(uh, th);
  a = b.nodeType;
  switch (a) {
    case 9:
    case 11:
      b = (b = b.documentElement) ? b.namespaceURI : lb(null, "");
      break;
    default:
      a = 8 === a ? b.parentNode : b, b = a.namespaceURI || null, a = a.tagName, b = lb(b, a);
  }
  E(uh);
  G(uh, b);
}
function zh() {
  E(uh);
  E(vh);
  E(wh);
}
function Ah(a) {
  xh(wh.current);
  var b = xh(uh.current);
  var c = lb(b, a.type);
  b !== c && (G(vh, a), G(uh, c));
}
function Bh(a) {
  vh.current === a && (E(uh), E(vh));
}
var L = Uf(0);
function Ch(a) {
  for (var b = a; null !== b; ) {
    if (13 === b.tag) {
      var c = b.memoizedState;
      if (null !== c && (c = c.dehydrated, null === c || "$?" === c.data || "$!" === c.data))
        return b;
    } else if (19 === b.tag && void 0 !== b.memoizedProps.revealOrder) {
      if (0 !== (b.flags & 128))
        return b;
    } else if (null !== b.child) {
      b.child.return = b;
      b = b.child;
      continue;
    }
    if (b === a)
      break;
    for (; null === b.sibling; ) {
      if (null === b.return || b.return === a)
        return null;
      b = b.return;
    }
    b.sibling.return = b.return;
    b = b.sibling;
  }
  return null;
}
var Dh = [];
function Eh() {
  for (var a = 0; a < Dh.length; a++)
    Dh[a]._workInProgressVersionPrimary = null;
  Dh.length = 0;
}
var Fh = ua.ReactCurrentDispatcher, Gh = ua.ReactCurrentBatchConfig, Hh = 0, M = null, N = null, O = null, Ih = false, Jh = false, Kh = 0, Lh = 0;
function P() {
  throw Error(p(321));
}
function Mh(a, b) {
  if (null === b)
    return false;
  for (var c = 0; c < b.length && c < a.length; c++)
    if (!He(a[c], b[c]))
      return false;
  return true;
}
function Nh(a, b, c, d, e, f2) {
  Hh = f2;
  M = b;
  b.memoizedState = null;
  b.updateQueue = null;
  b.lanes = 0;
  Fh.current = null === a || null === a.memoizedState ? Oh : Ph;
  a = c(d, e);
  if (Jh) {
    f2 = 0;
    do {
      Jh = false;
      Kh = 0;
      if (25 <= f2)
        throw Error(p(301));
      f2 += 1;
      O = N = null;
      b.updateQueue = null;
      Fh.current = Qh;
      a = c(d, e);
    } while (Jh);
  }
  Fh.current = Rh;
  b = null !== N && null !== N.next;
  Hh = 0;
  O = N = M = null;
  Ih = false;
  if (b)
    throw Error(p(300));
  return a;
}
function Sh() {
  var a = 0 !== Kh;
  Kh = 0;
  return a;
}
function Th() {
  var a = { memoizedState: null, baseState: null, baseQueue: null, queue: null, next: null };
  null === O ? M.memoizedState = O = a : O = O.next = a;
  return O;
}
function Uh() {
  if (null === N) {
    var a = M.alternate;
    a = null !== a ? a.memoizedState : null;
  } else
    a = N.next;
  var b = null === O ? M.memoizedState : O.next;
  if (null !== b)
    O = b, N = a;
  else {
    if (null === a)
      throw Error(p(310));
    N = a;
    a = { memoizedState: N.memoizedState, baseState: N.baseState, baseQueue: N.baseQueue, queue: N.queue, next: null };
    null === O ? M.memoizedState = O = a : O = O.next = a;
  }
  return O;
}
function Vh(a, b) {
  return "function" === typeof b ? b(a) : b;
}
function Wh(a) {
  var b = Uh(), c = b.queue;
  if (null === c)
    throw Error(p(311));
  c.lastRenderedReducer = a;
  var d = N, e = d.baseQueue, f2 = c.pending;
  if (null !== f2) {
    if (null !== e) {
      var g = e.next;
      e.next = f2.next;
      f2.next = g;
    }
    d.baseQueue = e = f2;
    c.pending = null;
  }
  if (null !== e) {
    f2 = e.next;
    d = d.baseState;
    var h = g = null, k2 = null, l2 = f2;
    do {
      var m2 = l2.lane;
      if ((Hh & m2) === m2)
        null !== k2 && (k2 = k2.next = { lane: 0, action: l2.action, hasEagerState: l2.hasEagerState, eagerState: l2.eagerState, next: null }), d = l2.hasEagerState ? l2.eagerState : a(d, l2.action);
      else {
        var q2 = {
          lane: m2,
          action: l2.action,
          hasEagerState: l2.hasEagerState,
          eagerState: l2.eagerState,
          next: null
        };
        null === k2 ? (h = k2 = q2, g = d) : k2 = k2.next = q2;
        M.lanes |= m2;
        rh |= m2;
      }
      l2 = l2.next;
    } while (null !== l2 && l2 !== f2);
    null === k2 ? g = d : k2.next = h;
    He(d, b.memoizedState) || (dh = true);
    b.memoizedState = d;
    b.baseState = g;
    b.baseQueue = k2;
    c.lastRenderedState = d;
  }
  a = c.interleaved;
  if (null !== a) {
    e = a;
    do
      f2 = e.lane, M.lanes |= f2, rh |= f2, e = e.next;
    while (e !== a);
  } else
    null === e && (c.lanes = 0);
  return [b.memoizedState, c.dispatch];
}
function Xh(a) {
  var b = Uh(), c = b.queue;
  if (null === c)
    throw Error(p(311));
  c.lastRenderedReducer = a;
  var d = c.dispatch, e = c.pending, f2 = b.memoizedState;
  if (null !== e) {
    c.pending = null;
    var g = e = e.next;
    do
      f2 = a(f2, g.action), g = g.next;
    while (g !== e);
    He(f2, b.memoizedState) || (dh = true);
    b.memoizedState = f2;
    null === b.baseQueue && (b.baseState = f2);
    c.lastRenderedState = f2;
  }
  return [f2, d];
}
function Yh() {
}
function Zh(a, b) {
  var c = M, d = Uh(), e = b(), f2 = !He(d.memoizedState, e);
  f2 && (d.memoizedState = e, dh = true);
  d = d.queue;
  $h(ai.bind(null, c, d, a), [a]);
  if (d.getSnapshot !== b || f2 || null !== O && O.memoizedState.tag & 1) {
    c.flags |= 2048;
    bi(9, ci.bind(null, c, d, e, b), void 0, null);
    if (null === Q)
      throw Error(p(349));
    0 !== (Hh & 30) || di(c, b, e);
  }
  return e;
}
function di(a, b, c) {
  a.flags |= 16384;
  a = { getSnapshot: b, value: c };
  b = M.updateQueue;
  null === b ? (b = { lastEffect: null, stores: null }, M.updateQueue = b, b.stores = [a]) : (c = b.stores, null === c ? b.stores = [a] : c.push(a));
}
function ci(a, b, c, d) {
  b.value = c;
  b.getSnapshot = d;
  ei(b) && fi(a);
}
function ai(a, b, c) {
  return c(function() {
    ei(b) && fi(a);
  });
}
function ei(a) {
  var b = a.getSnapshot;
  a = a.value;
  try {
    var c = b();
    return !He(a, c);
  } catch (d) {
    return true;
  }
}
function fi(a) {
  var b = ih(a, 1);
  null !== b && gi(b, a, 1, -1);
}
function hi(a) {
  var b = Th();
  "function" === typeof a && (a = a());
  b.memoizedState = b.baseState = a;
  a = { pending: null, interleaved: null, lanes: 0, dispatch: null, lastRenderedReducer: Vh, lastRenderedState: a };
  b.queue = a;
  a = a.dispatch = ii.bind(null, M, a);
  return [b.memoizedState, a];
}
function bi(a, b, c, d) {
  a = { tag: a, create: b, destroy: c, deps: d, next: null };
  b = M.updateQueue;
  null === b ? (b = { lastEffect: null, stores: null }, M.updateQueue = b, b.lastEffect = a.next = a) : (c = b.lastEffect, null === c ? b.lastEffect = a.next = a : (d = c.next, c.next = a, a.next = d, b.lastEffect = a));
  return a;
}
function ji() {
  return Uh().memoizedState;
}
function ki(a, b, c, d) {
  var e = Th();
  M.flags |= a;
  e.memoizedState = bi(1 | b, c, void 0, void 0 === d ? null : d);
}
function li(a, b, c, d) {
  var e = Uh();
  d = void 0 === d ? null : d;
  var f2 = void 0;
  if (null !== N) {
    var g = N.memoizedState;
    f2 = g.destroy;
    if (null !== d && Mh(d, g.deps)) {
      e.memoizedState = bi(b, c, f2, d);
      return;
    }
  }
  M.flags |= a;
  e.memoizedState = bi(1 | b, c, f2, d);
}
function mi(a, b) {
  return ki(8390656, 8, a, b);
}
function $h(a, b) {
  return li(2048, 8, a, b);
}
function ni(a, b) {
  return li(4, 2, a, b);
}
function oi(a, b) {
  return li(4, 4, a, b);
}
function pi(a, b) {
  if ("function" === typeof b)
    return a = a(), b(a), function() {
      b(null);
    };
  if (null !== b && void 0 !== b)
    return a = a(), b.current = a, function() {
      b.current = null;
    };
}
function qi(a, b, c) {
  c = null !== c && void 0 !== c ? c.concat([a]) : null;
  return li(4, 4, pi.bind(null, b, a), c);
}
function ri() {
}
function si(a, b) {
  var c = Uh();
  b = void 0 === b ? null : b;
  var d = c.memoizedState;
  if (null !== d && null !== b && Mh(b, d[1]))
    return d[0];
  c.memoizedState = [a, b];
  return a;
}
function ti(a, b) {
  var c = Uh();
  b = void 0 === b ? null : b;
  var d = c.memoizedState;
  if (null !== d && null !== b && Mh(b, d[1]))
    return d[0];
  a = a();
  c.memoizedState = [a, b];
  return a;
}
function ui(a, b, c) {
  if (0 === (Hh & 21))
    return a.baseState && (a.baseState = false, dh = true), a.memoizedState = c;
  He(c, b) || (c = yc(), M.lanes |= c, rh |= c, a.baseState = true);
  return b;
}
function vi(a, b) {
  var c = C;
  C = 0 !== c && 4 > c ? c : 4;
  a(true);
  var d = Gh.transition;
  Gh.transition = {};
  try {
    a(false), b();
  } finally {
    C = c, Gh.transition = d;
  }
}
function wi() {
  return Uh().memoizedState;
}
function xi(a, b, c) {
  var d = yi(a);
  c = { lane: d, action: c, hasEagerState: false, eagerState: null, next: null };
  if (zi(a))
    Ai(b, c);
  else if (c = hh(a, b, c, d), null !== c) {
    var e = R();
    gi(c, a, d, e);
    Bi(c, b, d);
  }
}
function ii(a, b, c) {
  var d = yi(a), e = { lane: d, action: c, hasEagerState: false, eagerState: null, next: null };
  if (zi(a))
    Ai(b, e);
  else {
    var f2 = a.alternate;
    if (0 === a.lanes && (null === f2 || 0 === f2.lanes) && (f2 = b.lastRenderedReducer, null !== f2))
      try {
        var g = b.lastRenderedState, h = f2(g, c);
        e.hasEagerState = true;
        e.eagerState = h;
        if (He(h, g)) {
          var k2 = b.interleaved;
          null === k2 ? (e.next = e, gh(b)) : (e.next = k2.next, k2.next = e);
          b.interleaved = e;
          return;
        }
      } catch (l2) {
      } finally {
      }
    c = hh(a, b, e, d);
    null !== c && (e = R(), gi(c, a, d, e), Bi(c, b, d));
  }
}
function zi(a) {
  var b = a.alternate;
  return a === M || null !== b && b === M;
}
function Ai(a, b) {
  Jh = Ih = true;
  var c = a.pending;
  null === c ? b.next = b : (b.next = c.next, c.next = b);
  a.pending = b;
}
function Bi(a, b, c) {
  if (0 !== (c & 4194240)) {
    var d = b.lanes;
    d &= a.pendingLanes;
    c |= d;
    b.lanes = c;
    Cc(a, c);
  }
}
var Rh = { readContext: eh, useCallback: P, useContext: P, useEffect: P, useImperativeHandle: P, useInsertionEffect: P, useLayoutEffect: P, useMemo: P, useReducer: P, useRef: P, useState: P, useDebugValue: P, useDeferredValue: P, useTransition: P, useMutableSource: P, useSyncExternalStore: P, useId: P, unstable_isNewReconciler: false }, Oh = { readContext: eh, useCallback: function(a, b) {
  Th().memoizedState = [a, void 0 === b ? null : b];
  return a;
}, useContext: eh, useEffect: mi, useImperativeHandle: function(a, b, c) {
  c = null !== c && void 0 !== c ? c.concat([a]) : null;
  return ki(
    4194308,
    4,
    pi.bind(null, b, a),
    c
  );
}, useLayoutEffect: function(a, b) {
  return ki(4194308, 4, a, b);
}, useInsertionEffect: function(a, b) {
  return ki(4, 2, a, b);
}, useMemo: function(a, b) {
  var c = Th();
  b = void 0 === b ? null : b;
  a = a();
  c.memoizedState = [a, b];
  return a;
}, useReducer: function(a, b, c) {
  var d = Th();
  b = void 0 !== c ? c(b) : b;
  d.memoizedState = d.baseState = b;
  a = { pending: null, interleaved: null, lanes: 0, dispatch: null, lastRenderedReducer: a, lastRenderedState: b };
  d.queue = a;
  a = a.dispatch = xi.bind(null, M, a);
  return [d.memoizedState, a];
}, useRef: function(a) {
  var b = Th();
  a = { current: a };
  return b.memoizedState = a;
}, useState: hi, useDebugValue: ri, useDeferredValue: function(a) {
  return Th().memoizedState = a;
}, useTransition: function() {
  var a = hi(false), b = a[0];
  a = vi.bind(null, a[1]);
  Th().memoizedState = a;
  return [b, a];
}, useMutableSource: function() {
}, useSyncExternalStore: function(a, b, c) {
  var d = M, e = Th();
  if (I) {
    if (void 0 === c)
      throw Error(p(407));
    c = c();
  } else {
    c = b();
    if (null === Q)
      throw Error(p(349));
    0 !== (Hh & 30) || di(d, b, c);
  }
  e.memoizedState = c;
  var f2 = { value: c, getSnapshot: b };
  e.queue = f2;
  mi(ai.bind(
    null,
    d,
    f2,
    a
  ), [a]);
  d.flags |= 2048;
  bi(9, ci.bind(null, d, f2, c, b), void 0, null);
  return c;
}, useId: function() {
  var a = Th(), b = Q.identifierPrefix;
  if (I) {
    var c = sg;
    var d = rg;
    c = (d & ~(1 << 32 - oc(d) - 1)).toString(32) + c;
    b = ":" + b + "R" + c;
    c = Kh++;
    0 < c && (b += "H" + c.toString(32));
    b += ":";
  } else
    c = Lh++, b = ":" + b + "r" + c.toString(32) + ":";
  return a.memoizedState = b;
}, unstable_isNewReconciler: false }, Ph = {
  readContext: eh,
  useCallback: si,
  useContext: eh,
  useEffect: $h,
  useImperativeHandle: qi,
  useInsertionEffect: ni,
  useLayoutEffect: oi,
  useMemo: ti,
  useReducer: Wh,
  useRef: ji,
  useState: function() {
    return Wh(Vh);
  },
  useDebugValue: ri,
  useDeferredValue: function(a) {
    var b = Uh();
    return ui(b, N.memoizedState, a);
  },
  useTransition: function() {
    var a = Wh(Vh)[0], b = Uh().memoizedState;
    return [a, b];
  },
  useMutableSource: Yh,
  useSyncExternalStore: Zh,
  useId: wi,
  unstable_isNewReconciler: false
}, Qh = { readContext: eh, useCallback: si, useContext: eh, useEffect: $h, useImperativeHandle: qi, useInsertionEffect: ni, useLayoutEffect: oi, useMemo: ti, useReducer: Xh, useRef: ji, useState: function() {
  return Xh(Vh);
}, useDebugValue: ri, useDeferredValue: function(a) {
  var b = Uh();
  return null === N ? b.memoizedState = a : ui(b, N.memoizedState, a);
}, useTransition: function() {
  var a = Xh(Vh)[0], b = Uh().memoizedState;
  return [a, b];
}, useMutableSource: Yh, useSyncExternalStore: Zh, useId: wi, unstable_isNewReconciler: false };
function Ci(a, b) {
  if (a && a.defaultProps) {
    b = A({}, b);
    a = a.defaultProps;
    for (var c in a)
      void 0 === b[c] && (b[c] = a[c]);
    return b;
  }
  return b;
}
function Di(a, b, c, d) {
  b = a.memoizedState;
  c = c(d, b);
  c = null === c || void 0 === c ? b : A({}, b, c);
  a.memoizedState = c;
  0 === a.lanes && (a.updateQueue.baseState = c);
}
var Ei = { isMounted: function(a) {
  return (a = a._reactInternals) ? Vb(a) === a : false;
}, enqueueSetState: function(a, b, c) {
  a = a._reactInternals;
  var d = R(), e = yi(a), f2 = mh(d, e);
  f2.payload = b;
  void 0 !== c && null !== c && (f2.callback = c);
  b = nh(a, f2, e);
  null !== b && (gi(b, a, e, d), oh(b, a, e));
}, enqueueReplaceState: function(a, b, c) {
  a = a._reactInternals;
  var d = R(), e = yi(a), f2 = mh(d, e);
  f2.tag = 1;
  f2.payload = b;
  void 0 !== c && null !== c && (f2.callback = c);
  b = nh(a, f2, e);
  null !== b && (gi(b, a, e, d), oh(b, a, e));
}, enqueueForceUpdate: function(a, b) {
  a = a._reactInternals;
  var c = R(), d = yi(a), e = mh(c, d);
  e.tag = 2;
  void 0 !== b && null !== b && (e.callback = b);
  b = nh(a, e, d);
  null !== b && (gi(b, a, d, c), oh(b, a, d));
} };
function Fi(a, b, c, d, e, f2, g) {
  a = a.stateNode;
  return "function" === typeof a.shouldComponentUpdate ? a.shouldComponentUpdate(d, f2, g) : b.prototype && b.prototype.isPureReactComponent ? !Ie(c, d) || !Ie(e, f2) : true;
}
function Gi(a, b, c) {
  var d = false, e = Vf;
  var f2 = b.contextType;
  "object" === typeof f2 && null !== f2 ? f2 = eh(f2) : (e = Zf(b) ? Xf : H.current, d = b.contextTypes, f2 = (d = null !== d && void 0 !== d) ? Yf(a, e) : Vf);
  b = new b(c, f2);
  a.memoizedState = null !== b.state && void 0 !== b.state ? b.state : null;
  b.updater = Ei;
  a.stateNode = b;
  b._reactInternals = a;
  d && (a = a.stateNode, a.__reactInternalMemoizedUnmaskedChildContext = e, a.__reactInternalMemoizedMaskedChildContext = f2);
  return b;
}
function Hi(a, b, c, d) {
  a = b.state;
  "function" === typeof b.componentWillReceiveProps && b.componentWillReceiveProps(c, d);
  "function" === typeof b.UNSAFE_componentWillReceiveProps && b.UNSAFE_componentWillReceiveProps(c, d);
  b.state !== a && Ei.enqueueReplaceState(b, b.state, null);
}
function Ii(a, b, c, d) {
  var e = a.stateNode;
  e.props = c;
  e.state = a.memoizedState;
  e.refs = {};
  kh(a);
  var f2 = b.contextType;
  "object" === typeof f2 && null !== f2 ? e.context = eh(f2) : (f2 = Zf(b) ? Xf : H.current, e.context = Yf(a, f2));
  e.state = a.memoizedState;
  f2 = b.getDerivedStateFromProps;
  "function" === typeof f2 && (Di(a, b, f2, c), e.state = a.memoizedState);
  "function" === typeof b.getDerivedStateFromProps || "function" === typeof e.getSnapshotBeforeUpdate || "function" !== typeof e.UNSAFE_componentWillMount && "function" !== typeof e.componentWillMount || (b = e.state, "function" === typeof e.componentWillMount && e.componentWillMount(), "function" === typeof e.UNSAFE_componentWillMount && e.UNSAFE_componentWillMount(), b !== e.state && Ei.enqueueReplaceState(e, e.state, null), qh(a, c, e, d), e.state = a.memoizedState);
  "function" === typeof e.componentDidMount && (a.flags |= 4194308);
}
function Ji(a, b) {
  try {
    var c = "", d = b;
    do
      c += Pa(d), d = d.return;
    while (d);
    var e = c;
  } catch (f2) {
    e = "\nError generating stack: " + f2.message + "\n" + f2.stack;
  }
  return { value: a, source: b, stack: e, digest: null };
}
function Ki(a, b, c) {
  return { value: a, source: null, stack: null != c ? c : null, digest: null != b ? b : null };
}
function Li(a, b) {
  try {
    console.error(b.value);
  } catch (c) {
    setTimeout(function() {
      throw c;
    });
  }
}
var Mi = "function" === typeof WeakMap ? WeakMap : Map;
function Ni(a, b, c) {
  c = mh(-1, c);
  c.tag = 3;
  c.payload = { element: null };
  var d = b.value;
  c.callback = function() {
    Oi || (Oi = true, Pi = d);
    Li(a, b);
  };
  return c;
}
function Qi(a, b, c) {
  c = mh(-1, c);
  c.tag = 3;
  var d = a.type.getDerivedStateFromError;
  if ("function" === typeof d) {
    var e = b.value;
    c.payload = function() {
      return d(e);
    };
    c.callback = function() {
      Li(a, b);
    };
  }
  var f2 = a.stateNode;
  null !== f2 && "function" === typeof f2.componentDidCatch && (c.callback = function() {
    Li(a, b);
    "function" !== typeof d && (null === Ri ? Ri = /* @__PURE__ */ new Set([this]) : Ri.add(this));
    var c2 = b.stack;
    this.componentDidCatch(b.value, { componentStack: null !== c2 ? c2 : "" });
  });
  return c;
}
function Si(a, b, c) {
  var d = a.pingCache;
  if (null === d) {
    d = a.pingCache = new Mi();
    var e = /* @__PURE__ */ new Set();
    d.set(b, e);
  } else
    e = d.get(b), void 0 === e && (e = /* @__PURE__ */ new Set(), d.set(b, e));
  e.has(c) || (e.add(c), a = Ti.bind(null, a, b, c), b.then(a, a));
}
function Ui(a) {
  do {
    var b;
    if (b = 13 === a.tag)
      b = a.memoizedState, b = null !== b ? null !== b.dehydrated ? true : false : true;
    if (b)
      return a;
    a = a.return;
  } while (null !== a);
  return null;
}
function Vi(a, b, c, d, e) {
  if (0 === (a.mode & 1))
    return a === b ? a.flags |= 65536 : (a.flags |= 128, c.flags |= 131072, c.flags &= -52805, 1 === c.tag && (null === c.alternate ? c.tag = 17 : (b = mh(-1, 1), b.tag = 2, nh(c, b, 1))), c.lanes |= 1), a;
  a.flags |= 65536;
  a.lanes = e;
  return a;
}
var Wi = ua.ReactCurrentOwner, dh = false;
function Xi(a, b, c, d) {
  b.child = null === a ? Vg(b, null, c, d) : Ug(b, a.child, c, d);
}
function Yi(a, b, c, d, e) {
  c = c.render;
  var f2 = b.ref;
  ch(b, e);
  d = Nh(a, b, c, d, f2, e);
  c = Sh();
  if (null !== a && !dh)
    return b.updateQueue = a.updateQueue, b.flags &= -2053, a.lanes &= ~e, Zi(a, b, e);
  I && c && vg(b);
  b.flags |= 1;
  Xi(a, b, d, e);
  return b.child;
}
function $i(a, b, c, d, e) {
  if (null === a) {
    var f2 = c.type;
    if ("function" === typeof f2 && !aj(f2) && void 0 === f2.defaultProps && null === c.compare && void 0 === c.defaultProps)
      return b.tag = 15, b.type = f2, bj(a, b, f2, d, e);
    a = Rg(c.type, null, d, b, b.mode, e);
    a.ref = b.ref;
    a.return = b;
    return b.child = a;
  }
  f2 = a.child;
  if (0 === (a.lanes & e)) {
    var g = f2.memoizedProps;
    c = c.compare;
    c = null !== c ? c : Ie;
    if (c(g, d) && a.ref === b.ref)
      return Zi(a, b, e);
  }
  b.flags |= 1;
  a = Pg(f2, d);
  a.ref = b.ref;
  a.return = b;
  return b.child = a;
}
function bj(a, b, c, d, e) {
  if (null !== a) {
    var f2 = a.memoizedProps;
    if (Ie(f2, d) && a.ref === b.ref)
      if (dh = false, b.pendingProps = d = f2, 0 !== (a.lanes & e))
        0 !== (a.flags & 131072) && (dh = true);
      else
        return b.lanes = a.lanes, Zi(a, b, e);
  }
  return cj(a, b, c, d, e);
}
function dj(a, b, c) {
  var d = b.pendingProps, e = d.children, f2 = null !== a ? a.memoizedState : null;
  if ("hidden" === d.mode)
    if (0 === (b.mode & 1))
      b.memoizedState = { baseLanes: 0, cachePool: null, transitions: null }, G(ej, fj), fj |= c;
    else {
      if (0 === (c & 1073741824))
        return a = null !== f2 ? f2.baseLanes | c : c, b.lanes = b.childLanes = 1073741824, b.memoizedState = { baseLanes: a, cachePool: null, transitions: null }, b.updateQueue = null, G(ej, fj), fj |= a, null;
      b.memoizedState = { baseLanes: 0, cachePool: null, transitions: null };
      d = null !== f2 ? f2.baseLanes : c;
      G(ej, fj);
      fj |= d;
    }
  else
    null !== f2 ? (d = f2.baseLanes | c, b.memoizedState = null) : d = c, G(ej, fj), fj |= d;
  Xi(a, b, e, c);
  return b.child;
}
function gj(a, b) {
  var c = b.ref;
  if (null === a && null !== c || null !== a && a.ref !== c)
    b.flags |= 512, b.flags |= 2097152;
}
function cj(a, b, c, d, e) {
  var f2 = Zf(c) ? Xf : H.current;
  f2 = Yf(b, f2);
  ch(b, e);
  c = Nh(a, b, c, d, f2, e);
  d = Sh();
  if (null !== a && !dh)
    return b.updateQueue = a.updateQueue, b.flags &= -2053, a.lanes &= ~e, Zi(a, b, e);
  I && d && vg(b);
  b.flags |= 1;
  Xi(a, b, c, e);
  return b.child;
}
function hj(a, b, c, d, e) {
  if (Zf(c)) {
    var f2 = true;
    cg(b);
  } else
    f2 = false;
  ch(b, e);
  if (null === b.stateNode)
    ij(a, b), Gi(b, c, d), Ii(b, c, d, e), d = true;
  else if (null === a) {
    var g = b.stateNode, h = b.memoizedProps;
    g.props = h;
    var k2 = g.context, l2 = c.contextType;
    "object" === typeof l2 && null !== l2 ? l2 = eh(l2) : (l2 = Zf(c) ? Xf : H.current, l2 = Yf(b, l2));
    var m2 = c.getDerivedStateFromProps, q2 = "function" === typeof m2 || "function" === typeof g.getSnapshotBeforeUpdate;
    q2 || "function" !== typeof g.UNSAFE_componentWillReceiveProps && "function" !== typeof g.componentWillReceiveProps || (h !== d || k2 !== l2) && Hi(b, g, d, l2);
    jh = false;
    var r2 = b.memoizedState;
    g.state = r2;
    qh(b, d, g, e);
    k2 = b.memoizedState;
    h !== d || r2 !== k2 || Wf.current || jh ? ("function" === typeof m2 && (Di(b, c, m2, d), k2 = b.memoizedState), (h = jh || Fi(b, c, h, d, r2, k2, l2)) ? (q2 || "function" !== typeof g.UNSAFE_componentWillMount && "function" !== typeof g.componentWillMount || ("function" === typeof g.componentWillMount && g.componentWillMount(), "function" === typeof g.UNSAFE_componentWillMount && g.UNSAFE_componentWillMount()), "function" === typeof g.componentDidMount && (b.flags |= 4194308)) : ("function" === typeof g.componentDidMount && (b.flags |= 4194308), b.memoizedProps = d, b.memoizedState = k2), g.props = d, g.state = k2, g.context = l2, d = h) : ("function" === typeof g.componentDidMount && (b.flags |= 4194308), d = false);
  } else {
    g = b.stateNode;
    lh(a, b);
    h = b.memoizedProps;
    l2 = b.type === b.elementType ? h : Ci(b.type, h);
    g.props = l2;
    q2 = b.pendingProps;
    r2 = g.context;
    k2 = c.contextType;
    "object" === typeof k2 && null !== k2 ? k2 = eh(k2) : (k2 = Zf(c) ? Xf : H.current, k2 = Yf(b, k2));
    var y2 = c.getDerivedStateFromProps;
    (m2 = "function" === typeof y2 || "function" === typeof g.getSnapshotBeforeUpdate) || "function" !== typeof g.UNSAFE_componentWillReceiveProps && "function" !== typeof g.componentWillReceiveProps || (h !== q2 || r2 !== k2) && Hi(b, g, d, k2);
    jh = false;
    r2 = b.memoizedState;
    g.state = r2;
    qh(b, d, g, e);
    var n2 = b.memoizedState;
    h !== q2 || r2 !== n2 || Wf.current || jh ? ("function" === typeof y2 && (Di(b, c, y2, d), n2 = b.memoizedState), (l2 = jh || Fi(b, c, l2, d, r2, n2, k2) || false) ? (m2 || "function" !== typeof g.UNSAFE_componentWillUpdate && "function" !== typeof g.componentWillUpdate || ("function" === typeof g.componentWillUpdate && g.componentWillUpdate(d, n2, k2), "function" === typeof g.UNSAFE_componentWillUpdate && g.UNSAFE_componentWillUpdate(d, n2, k2)), "function" === typeof g.componentDidUpdate && (b.flags |= 4), "function" === typeof g.getSnapshotBeforeUpdate && (b.flags |= 1024)) : ("function" !== typeof g.componentDidUpdate || h === a.memoizedProps && r2 === a.memoizedState || (b.flags |= 4), "function" !== typeof g.getSnapshotBeforeUpdate || h === a.memoizedProps && r2 === a.memoizedState || (b.flags |= 1024), b.memoizedProps = d, b.memoizedState = n2), g.props = d, g.state = n2, g.context = k2, d = l2) : ("function" !== typeof g.componentDidUpdate || h === a.memoizedProps && r2 === a.memoizedState || (b.flags |= 4), "function" !== typeof g.getSnapshotBeforeUpdate || h === a.memoizedProps && r2 === a.memoizedState || (b.flags |= 1024), d = false);
  }
  return jj(a, b, c, d, f2, e);
}
function jj(a, b, c, d, e, f2) {
  gj(a, b);
  var g = 0 !== (b.flags & 128);
  if (!d && !g)
    return e && dg(b, c, false), Zi(a, b, f2);
  d = b.stateNode;
  Wi.current = b;
  var h = g && "function" !== typeof c.getDerivedStateFromError ? null : d.render();
  b.flags |= 1;
  null !== a && g ? (b.child = Ug(b, a.child, null, f2), b.child = Ug(b, null, h, f2)) : Xi(a, b, h, f2);
  b.memoizedState = d.state;
  e && dg(b, c, true);
  return b.child;
}
function kj(a) {
  var b = a.stateNode;
  b.pendingContext ? ag(a, b.pendingContext, b.pendingContext !== b.context) : b.context && ag(a, b.context, false);
  yh(a, b.containerInfo);
}
function lj(a, b, c, d, e) {
  Ig();
  Jg(e);
  b.flags |= 256;
  Xi(a, b, c, d);
  return b.child;
}
var mj = { dehydrated: null, treeContext: null, retryLane: 0 };
function nj(a) {
  return { baseLanes: a, cachePool: null, transitions: null };
}
function oj(a, b, c) {
  var d = b.pendingProps, e = L.current, f2 = false, g = 0 !== (b.flags & 128), h;
  (h = g) || (h = null !== a && null === a.memoizedState ? false : 0 !== (e & 2));
  if (h)
    f2 = true, b.flags &= -129;
  else if (null === a || null !== a.memoizedState)
    e |= 1;
  G(L, e & 1);
  if (null === a) {
    Eg(b);
    a = b.memoizedState;
    if (null !== a && (a = a.dehydrated, null !== a))
      return 0 === (b.mode & 1) ? b.lanes = 1 : "$!" === a.data ? b.lanes = 8 : b.lanes = 1073741824, null;
    g = d.children;
    a = d.fallback;
    return f2 ? (d = b.mode, f2 = b.child, g = { mode: "hidden", children: g }, 0 === (d & 1) && null !== f2 ? (f2.childLanes = 0, f2.pendingProps = g) : f2 = pj(g, d, 0, null), a = Tg(a, d, c, null), f2.return = b, a.return = b, f2.sibling = a, b.child = f2, b.child.memoizedState = nj(c), b.memoizedState = mj, a) : qj(b, g);
  }
  e = a.memoizedState;
  if (null !== e && (h = e.dehydrated, null !== h))
    return rj(a, b, g, d, h, e, c);
  if (f2) {
    f2 = d.fallback;
    g = b.mode;
    e = a.child;
    h = e.sibling;
    var k2 = { mode: "hidden", children: d.children };
    0 === (g & 1) && b.child !== e ? (d = b.child, d.childLanes = 0, d.pendingProps = k2, b.deletions = null) : (d = Pg(e, k2), d.subtreeFlags = e.subtreeFlags & 14680064);
    null !== h ? f2 = Pg(h, f2) : (f2 = Tg(f2, g, c, null), f2.flags |= 2);
    f2.return = b;
    d.return = b;
    d.sibling = f2;
    b.child = d;
    d = f2;
    f2 = b.child;
    g = a.child.memoizedState;
    g = null === g ? nj(c) : { baseLanes: g.baseLanes | c, cachePool: null, transitions: g.transitions };
    f2.memoizedState = g;
    f2.childLanes = a.childLanes & ~c;
    b.memoizedState = mj;
    return d;
  }
  f2 = a.child;
  a = f2.sibling;
  d = Pg(f2, { mode: "visible", children: d.children });
  0 === (b.mode & 1) && (d.lanes = c);
  d.return = b;
  d.sibling = null;
  null !== a && (c = b.deletions, null === c ? (b.deletions = [a], b.flags |= 16) : c.push(a));
  b.child = d;
  b.memoizedState = null;
  return d;
}
function qj(a, b) {
  b = pj({ mode: "visible", children: b }, a.mode, 0, null);
  b.return = a;
  return a.child = b;
}
function sj(a, b, c, d) {
  null !== d && Jg(d);
  Ug(b, a.child, null, c);
  a = qj(b, b.pendingProps.children);
  a.flags |= 2;
  b.memoizedState = null;
  return a;
}
function rj(a, b, c, d, e, f2, g) {
  if (c) {
    if (b.flags & 256)
      return b.flags &= -257, d = Ki(Error(p(422))), sj(a, b, g, d);
    if (null !== b.memoizedState)
      return b.child = a.child, b.flags |= 128, null;
    f2 = d.fallback;
    e = b.mode;
    d = pj({ mode: "visible", children: d.children }, e, 0, null);
    f2 = Tg(f2, e, g, null);
    f2.flags |= 2;
    d.return = b;
    f2.return = b;
    d.sibling = f2;
    b.child = d;
    0 !== (b.mode & 1) && Ug(b, a.child, null, g);
    b.child.memoizedState = nj(g);
    b.memoizedState = mj;
    return f2;
  }
  if (0 === (b.mode & 1))
    return sj(a, b, g, null);
  if ("$!" === e.data) {
    d = e.nextSibling && e.nextSibling.dataset;
    if (d)
      var h = d.dgst;
    d = h;
    f2 = Error(p(419));
    d = Ki(f2, d, void 0);
    return sj(a, b, g, d);
  }
  h = 0 !== (g & a.childLanes);
  if (dh || h) {
    d = Q;
    if (null !== d) {
      switch (g & -g) {
        case 4:
          e = 2;
          break;
        case 16:
          e = 8;
          break;
        case 64:
        case 128:
        case 256:
        case 512:
        case 1024:
        case 2048:
        case 4096:
        case 8192:
        case 16384:
        case 32768:
        case 65536:
        case 131072:
        case 262144:
        case 524288:
        case 1048576:
        case 2097152:
        case 4194304:
        case 8388608:
        case 16777216:
        case 33554432:
        case 67108864:
          e = 32;
          break;
        case 536870912:
          e = 268435456;
          break;
        default:
          e = 0;
      }
      e = 0 !== (e & (d.suspendedLanes | g)) ? 0 : e;
      0 !== e && e !== f2.retryLane && (f2.retryLane = e, ih(a, e), gi(d, a, e, -1));
    }
    tj();
    d = Ki(Error(p(421)));
    return sj(a, b, g, d);
  }
  if ("$?" === e.data)
    return b.flags |= 128, b.child = a.child, b = uj.bind(null, a), e._reactRetry = b, null;
  a = f2.treeContext;
  yg = Lf(e.nextSibling);
  xg = b;
  I = true;
  zg = null;
  null !== a && (og[pg++] = rg, og[pg++] = sg, og[pg++] = qg, rg = a.id, sg = a.overflow, qg = b);
  b = qj(b, d.children);
  b.flags |= 4096;
  return b;
}
function vj(a, b, c) {
  a.lanes |= b;
  var d = a.alternate;
  null !== d && (d.lanes |= b);
  bh(a.return, b, c);
}
function wj(a, b, c, d, e) {
  var f2 = a.memoizedState;
  null === f2 ? a.memoizedState = { isBackwards: b, rendering: null, renderingStartTime: 0, last: d, tail: c, tailMode: e } : (f2.isBackwards = b, f2.rendering = null, f2.renderingStartTime = 0, f2.last = d, f2.tail = c, f2.tailMode = e);
}
function xj(a, b, c) {
  var d = b.pendingProps, e = d.revealOrder, f2 = d.tail;
  Xi(a, b, d.children, c);
  d = L.current;
  if (0 !== (d & 2))
    d = d & 1 | 2, b.flags |= 128;
  else {
    if (null !== a && 0 !== (a.flags & 128))
      a:
        for (a = b.child; null !== a; ) {
          if (13 === a.tag)
            null !== a.memoizedState && vj(a, c, b);
          else if (19 === a.tag)
            vj(a, c, b);
          else if (null !== a.child) {
            a.child.return = a;
            a = a.child;
            continue;
          }
          if (a === b)
            break a;
          for (; null === a.sibling; ) {
            if (null === a.return || a.return === b)
              break a;
            a = a.return;
          }
          a.sibling.return = a.return;
          a = a.sibling;
        }
    d &= 1;
  }
  G(L, d);
  if (0 === (b.mode & 1))
    b.memoizedState = null;
  else
    switch (e) {
      case "forwards":
        c = b.child;
        for (e = null; null !== c; )
          a = c.alternate, null !== a && null === Ch(a) && (e = c), c = c.sibling;
        c = e;
        null === c ? (e = b.child, b.child = null) : (e = c.sibling, c.sibling = null);
        wj(b, false, e, c, f2);
        break;
      case "backwards":
        c = null;
        e = b.child;
        for (b.child = null; null !== e; ) {
          a = e.alternate;
          if (null !== a && null === Ch(a)) {
            b.child = e;
            break;
          }
          a = e.sibling;
          e.sibling = c;
          c = e;
          e = a;
        }
        wj(b, true, c, null, f2);
        break;
      case "together":
        wj(b, false, null, null, void 0);
        break;
      default:
        b.memoizedState = null;
    }
  return b.child;
}
function ij(a, b) {
  0 === (b.mode & 1) && null !== a && (a.alternate = null, b.alternate = null, b.flags |= 2);
}
function Zi(a, b, c) {
  null !== a && (b.dependencies = a.dependencies);
  rh |= b.lanes;
  if (0 === (c & b.childLanes))
    return null;
  if (null !== a && b.child !== a.child)
    throw Error(p(153));
  if (null !== b.child) {
    a = b.child;
    c = Pg(a, a.pendingProps);
    b.child = c;
    for (c.return = b; null !== a.sibling; )
      a = a.sibling, c = c.sibling = Pg(a, a.pendingProps), c.return = b;
    c.sibling = null;
  }
  return b.child;
}
function yj(a, b, c) {
  switch (b.tag) {
    case 3:
      kj(b);
      Ig();
      break;
    case 5:
      Ah(b);
      break;
    case 1:
      Zf(b.type) && cg(b);
      break;
    case 4:
      yh(b, b.stateNode.containerInfo);
      break;
    case 10:
      var d = b.type._context, e = b.memoizedProps.value;
      G(Wg, d._currentValue);
      d._currentValue = e;
      break;
    case 13:
      d = b.memoizedState;
      if (null !== d) {
        if (null !== d.dehydrated)
          return G(L, L.current & 1), b.flags |= 128, null;
        if (0 !== (c & b.child.childLanes))
          return oj(a, b, c);
        G(L, L.current & 1);
        a = Zi(a, b, c);
        return null !== a ? a.sibling : null;
      }
      G(L, L.current & 1);
      break;
    case 19:
      d = 0 !== (c & b.childLanes);
      if (0 !== (a.flags & 128)) {
        if (d)
          return xj(a, b, c);
        b.flags |= 128;
      }
      e = b.memoizedState;
      null !== e && (e.rendering = null, e.tail = null, e.lastEffect = null);
      G(L, L.current);
      if (d)
        break;
      else
        return null;
    case 22:
    case 23:
      return b.lanes = 0, dj(a, b, c);
  }
  return Zi(a, b, c);
}
var zj, Aj, Bj, Cj;
zj = function(a, b) {
  for (var c = b.child; null !== c; ) {
    if (5 === c.tag || 6 === c.tag)
      a.appendChild(c.stateNode);
    else if (4 !== c.tag && null !== c.child) {
      c.child.return = c;
      c = c.child;
      continue;
    }
    if (c === b)
      break;
    for (; null === c.sibling; ) {
      if (null === c.return || c.return === b)
        return;
      c = c.return;
    }
    c.sibling.return = c.return;
    c = c.sibling;
  }
};
Aj = function() {
};
Bj = function(a, b, c, d) {
  var e = a.memoizedProps;
  if (e !== d) {
    a = b.stateNode;
    xh(uh.current);
    var f2 = null;
    switch (c) {
      case "input":
        e = Ya(a, e);
        d = Ya(a, d);
        f2 = [];
        break;
      case "select":
        e = A({}, e, { value: void 0 });
        d = A({}, d, { value: void 0 });
        f2 = [];
        break;
      case "textarea":
        e = gb(a, e);
        d = gb(a, d);
        f2 = [];
        break;
      default:
        "function" !== typeof e.onClick && "function" === typeof d.onClick && (a.onclick = Bf);
    }
    ub(c, d);
    var g;
    c = null;
    for (l2 in e)
      if (!d.hasOwnProperty(l2) && e.hasOwnProperty(l2) && null != e[l2])
        if ("style" === l2) {
          var h = e[l2];
          for (g in h)
            h.hasOwnProperty(g) && (c || (c = {}), c[g] = "");
        } else
          "dangerouslySetInnerHTML" !== l2 && "children" !== l2 && "suppressContentEditableWarning" !== l2 && "suppressHydrationWarning" !== l2 && "autoFocus" !== l2 && (ea.hasOwnProperty(l2) ? f2 || (f2 = []) : (f2 = f2 || []).push(l2, null));
    for (l2 in d) {
      var k2 = d[l2];
      h = null != e ? e[l2] : void 0;
      if (d.hasOwnProperty(l2) && k2 !== h && (null != k2 || null != h))
        if ("style" === l2)
          if (h) {
            for (g in h)
              !h.hasOwnProperty(g) || k2 && k2.hasOwnProperty(g) || (c || (c = {}), c[g] = "");
            for (g in k2)
              k2.hasOwnProperty(g) && h[g] !== k2[g] && (c || (c = {}), c[g] = k2[g]);
          } else
            c || (f2 || (f2 = []), f2.push(
              l2,
              c
            )), c = k2;
        else
          "dangerouslySetInnerHTML" === l2 ? (k2 = k2 ? k2.__html : void 0, h = h ? h.__html : void 0, null != k2 && h !== k2 && (f2 = f2 || []).push(l2, k2)) : "children" === l2 ? "string" !== typeof k2 && "number" !== typeof k2 || (f2 = f2 || []).push(l2, "" + k2) : "suppressContentEditableWarning" !== l2 && "suppressHydrationWarning" !== l2 && (ea.hasOwnProperty(l2) ? (null != k2 && "onScroll" === l2 && D("scroll", a), f2 || h === k2 || (f2 = [])) : (f2 = f2 || []).push(l2, k2));
    }
    c && (f2 = f2 || []).push("style", c);
    var l2 = f2;
    if (b.updateQueue = l2)
      b.flags |= 4;
  }
};
Cj = function(a, b, c, d) {
  c !== d && (b.flags |= 4);
};
function Dj(a, b) {
  if (!I)
    switch (a.tailMode) {
      case "hidden":
        b = a.tail;
        for (var c = null; null !== b; )
          null !== b.alternate && (c = b), b = b.sibling;
        null === c ? a.tail = null : c.sibling = null;
        break;
      case "collapsed":
        c = a.tail;
        for (var d = null; null !== c; )
          null !== c.alternate && (d = c), c = c.sibling;
        null === d ? b || null === a.tail ? a.tail = null : a.tail.sibling = null : d.sibling = null;
    }
}
function S(a) {
  var b = null !== a.alternate && a.alternate.child === a.child, c = 0, d = 0;
  if (b)
    for (var e = a.child; null !== e; )
      c |= e.lanes | e.childLanes, d |= e.subtreeFlags & 14680064, d |= e.flags & 14680064, e.return = a, e = e.sibling;
  else
    for (e = a.child; null !== e; )
      c |= e.lanes | e.childLanes, d |= e.subtreeFlags, d |= e.flags, e.return = a, e = e.sibling;
  a.subtreeFlags |= d;
  a.childLanes = c;
  return b;
}
function Ej(a, b, c) {
  var d = b.pendingProps;
  wg(b);
  switch (b.tag) {
    case 2:
    case 16:
    case 15:
    case 0:
    case 11:
    case 7:
    case 8:
    case 12:
    case 9:
    case 14:
      return S(b), null;
    case 1:
      return Zf(b.type) && $f(), S(b), null;
    case 3:
      d = b.stateNode;
      zh();
      E(Wf);
      E(H);
      Eh();
      d.pendingContext && (d.context = d.pendingContext, d.pendingContext = null);
      if (null === a || null === a.child)
        Gg(b) ? b.flags |= 4 : null === a || a.memoizedState.isDehydrated && 0 === (b.flags & 256) || (b.flags |= 1024, null !== zg && (Fj(zg), zg = null));
      Aj(a, b);
      S(b);
      return null;
    case 5:
      Bh(b);
      var e = xh(wh.current);
      c = b.type;
      if (null !== a && null != b.stateNode)
        Bj(a, b, c, d, e), a.ref !== b.ref && (b.flags |= 512, b.flags |= 2097152);
      else {
        if (!d) {
          if (null === b.stateNode)
            throw Error(p(166));
          S(b);
          return null;
        }
        a = xh(uh.current);
        if (Gg(b)) {
          d = b.stateNode;
          c = b.type;
          var f2 = b.memoizedProps;
          d[Of] = b;
          d[Pf] = f2;
          a = 0 !== (b.mode & 1);
          switch (c) {
            case "dialog":
              D("cancel", d);
              D("close", d);
              break;
            case "iframe":
            case "object":
            case "embed":
              D("load", d);
              break;
            case "video":
            case "audio":
              for (e = 0; e < lf.length; e++)
                D(lf[e], d);
              break;
            case "source":
              D("error", d);
              break;
            case "img":
            case "image":
            case "link":
              D(
                "error",
                d
              );
              D("load", d);
              break;
            case "details":
              D("toggle", d);
              break;
            case "input":
              Za(d, f2);
              D("invalid", d);
              break;
            case "select":
              d._wrapperState = { wasMultiple: !!f2.multiple };
              D("invalid", d);
              break;
            case "textarea":
              hb(d, f2), D("invalid", d);
          }
          ub(c, f2);
          e = null;
          for (var g in f2)
            if (f2.hasOwnProperty(g)) {
              var h = f2[g];
              "children" === g ? "string" === typeof h ? d.textContent !== h && (true !== f2.suppressHydrationWarning && Af(d.textContent, h, a), e = ["children", h]) : "number" === typeof h && d.textContent !== "" + h && (true !== f2.suppressHydrationWarning && Af(
                d.textContent,
                h,
                a
              ), e = ["children", "" + h]) : ea.hasOwnProperty(g) && null != h && "onScroll" === g && D("scroll", d);
            }
          switch (c) {
            case "input":
              Va(d);
              db(d, f2, true);
              break;
            case "textarea":
              Va(d);
              jb(d);
              break;
            case "select":
            case "option":
              break;
            default:
              "function" === typeof f2.onClick && (d.onclick = Bf);
          }
          d = e;
          b.updateQueue = d;
          null !== d && (b.flags |= 4);
        } else {
          g = 9 === e.nodeType ? e : e.ownerDocument;
          "http://www.w3.org/1999/xhtml" === a && (a = kb(c));
          "http://www.w3.org/1999/xhtml" === a ? "script" === c ? (a = g.createElement("div"), a.innerHTML = "<script><\/script>", a = a.removeChild(a.firstChild)) : "string" === typeof d.is ? a = g.createElement(c, { is: d.is }) : (a = g.createElement(c), "select" === c && (g = a, d.multiple ? g.multiple = true : d.size && (g.size = d.size))) : a = g.createElementNS(a, c);
          a[Of] = b;
          a[Pf] = d;
          zj(a, b, false, false);
          b.stateNode = a;
          a: {
            g = vb(c, d);
            switch (c) {
              case "dialog":
                D("cancel", a);
                D("close", a);
                e = d;
                break;
              case "iframe":
              case "object":
              case "embed":
                D("load", a);
                e = d;
                break;
              case "video":
              case "audio":
                for (e = 0; e < lf.length; e++)
                  D(lf[e], a);
                e = d;
                break;
              case "source":
                D("error", a);
                e = d;
                break;
              case "img":
              case "image":
              case "link":
                D(
                  "error",
                  a
                );
                D("load", a);
                e = d;
                break;
              case "details":
                D("toggle", a);
                e = d;
                break;
              case "input":
                Za(a, d);
                e = Ya(a, d);
                D("invalid", a);
                break;
              case "option":
                e = d;
                break;
              case "select":
                a._wrapperState = { wasMultiple: !!d.multiple };
                e = A({}, d, { value: void 0 });
                D("invalid", a);
                break;
              case "textarea":
                hb(a, d);
                e = gb(a, d);
                D("invalid", a);
                break;
              default:
                e = d;
            }
            ub(c, e);
            h = e;
            for (f2 in h)
              if (h.hasOwnProperty(f2)) {
                var k2 = h[f2];
                "style" === f2 ? sb(a, k2) : "dangerouslySetInnerHTML" === f2 ? (k2 = k2 ? k2.__html : void 0, null != k2 && nb(a, k2)) : "children" === f2 ? "string" === typeof k2 ? ("textarea" !== c || "" !== k2) && ob(a, k2) : "number" === typeof k2 && ob(a, "" + k2) : "suppressContentEditableWarning" !== f2 && "suppressHydrationWarning" !== f2 && "autoFocus" !== f2 && (ea.hasOwnProperty(f2) ? null != k2 && "onScroll" === f2 && D("scroll", a) : null != k2 && ta(a, f2, k2, g));
              }
            switch (c) {
              case "input":
                Va(a);
                db(a, d, false);
                break;
              case "textarea":
                Va(a);
                jb(a);
                break;
              case "option":
                null != d.value && a.setAttribute("value", "" + Sa(d.value));
                break;
              case "select":
                a.multiple = !!d.multiple;
                f2 = d.value;
                null != f2 ? fb(a, !!d.multiple, f2, false) : null != d.defaultValue && fb(
                  a,
                  !!d.multiple,
                  d.defaultValue,
                  true
                );
                break;
              default:
                "function" === typeof e.onClick && (a.onclick = Bf);
            }
            switch (c) {
              case "button":
              case "input":
              case "select":
              case "textarea":
                d = !!d.autoFocus;
                break a;
              case "img":
                d = true;
                break a;
              default:
                d = false;
            }
          }
          d && (b.flags |= 4);
        }
        null !== b.ref && (b.flags |= 512, b.flags |= 2097152);
      }
      S(b);
      return null;
    case 6:
      if (a && null != b.stateNode)
        Cj(a, b, a.memoizedProps, d);
      else {
        if ("string" !== typeof d && null === b.stateNode)
          throw Error(p(166));
        c = xh(wh.current);
        xh(uh.current);
        if (Gg(b)) {
          d = b.stateNode;
          c = b.memoizedProps;
          d[Of] = b;
          if (f2 = d.nodeValue !== c) {
            if (a = xg, null !== a)
              switch (a.tag) {
                case 3:
                  Af(d.nodeValue, c, 0 !== (a.mode & 1));
                  break;
                case 5:
                  true !== a.memoizedProps.suppressHydrationWarning && Af(d.nodeValue, c, 0 !== (a.mode & 1));
              }
          }
          f2 && (b.flags |= 4);
        } else
          d = (9 === c.nodeType ? c : c.ownerDocument).createTextNode(d), d[Of] = b, b.stateNode = d;
      }
      S(b);
      return null;
    case 13:
      E(L);
      d = b.memoizedState;
      if (null === a || null !== a.memoizedState && null !== a.memoizedState.dehydrated) {
        if (I && null !== yg && 0 !== (b.mode & 1) && 0 === (b.flags & 128))
          Hg(), Ig(), b.flags |= 98560, f2 = false;
        else if (f2 = Gg(b), null !== d && null !== d.dehydrated) {
          if (null === a) {
            if (!f2)
              throw Error(p(318));
            f2 = b.memoizedState;
            f2 = null !== f2 ? f2.dehydrated : null;
            if (!f2)
              throw Error(p(317));
            f2[Of] = b;
          } else
            Ig(), 0 === (b.flags & 128) && (b.memoizedState = null), b.flags |= 4;
          S(b);
          f2 = false;
        } else
          null !== zg && (Fj(zg), zg = null), f2 = true;
        if (!f2)
          return b.flags & 65536 ? b : null;
      }
      if (0 !== (b.flags & 128))
        return b.lanes = c, b;
      d = null !== d;
      d !== (null !== a && null !== a.memoizedState) && d && (b.child.flags |= 8192, 0 !== (b.mode & 1) && (null === a || 0 !== (L.current & 1) ? 0 === T && (T = 3) : tj()));
      null !== b.updateQueue && (b.flags |= 4);
      S(b);
      return null;
    case 4:
      return zh(), Aj(a, b), null === a && sf(b.stateNode.containerInfo), S(b), null;
    case 10:
      return ah(b.type._context), S(b), null;
    case 17:
      return Zf(b.type) && $f(), S(b), null;
    case 19:
      E(L);
      f2 = b.memoizedState;
      if (null === f2)
        return S(b), null;
      d = 0 !== (b.flags & 128);
      g = f2.rendering;
      if (null === g)
        if (d)
          Dj(f2, false);
        else {
          if (0 !== T || null !== a && 0 !== (a.flags & 128))
            for (a = b.child; null !== a; ) {
              g = Ch(a);
              if (null !== g) {
                b.flags |= 128;
                Dj(f2, false);
                d = g.updateQueue;
                null !== d && (b.updateQueue = d, b.flags |= 4);
                b.subtreeFlags = 0;
                d = c;
                for (c = b.child; null !== c; )
                  f2 = c, a = d, f2.flags &= 14680066, g = f2.alternate, null === g ? (f2.childLanes = 0, f2.lanes = a, f2.child = null, f2.subtreeFlags = 0, f2.memoizedProps = null, f2.memoizedState = null, f2.updateQueue = null, f2.dependencies = null, f2.stateNode = null) : (f2.childLanes = g.childLanes, f2.lanes = g.lanes, f2.child = g.child, f2.subtreeFlags = 0, f2.deletions = null, f2.memoizedProps = g.memoizedProps, f2.memoizedState = g.memoizedState, f2.updateQueue = g.updateQueue, f2.type = g.type, a = g.dependencies, f2.dependencies = null === a ? null : { lanes: a.lanes, firstContext: a.firstContext }), c = c.sibling;
                G(L, L.current & 1 | 2);
                return b.child;
              }
              a = a.sibling;
            }
          null !== f2.tail && B() > Gj && (b.flags |= 128, d = true, Dj(f2, false), b.lanes = 4194304);
        }
      else {
        if (!d)
          if (a = Ch(g), null !== a) {
            if (b.flags |= 128, d = true, c = a.updateQueue, null !== c && (b.updateQueue = c, b.flags |= 4), Dj(f2, true), null === f2.tail && "hidden" === f2.tailMode && !g.alternate && !I)
              return S(b), null;
          } else
            2 * B() - f2.renderingStartTime > Gj && 1073741824 !== c && (b.flags |= 128, d = true, Dj(f2, false), b.lanes = 4194304);
        f2.isBackwards ? (g.sibling = b.child, b.child = g) : (c = f2.last, null !== c ? c.sibling = g : b.child = g, f2.last = g);
      }
      if (null !== f2.tail)
        return b = f2.tail, f2.rendering = b, f2.tail = b.sibling, f2.renderingStartTime = B(), b.sibling = null, c = L.current, G(L, d ? c & 1 | 2 : c & 1), b;
      S(b);
      return null;
    case 22:
    case 23:
      return Hj(), d = null !== b.memoizedState, null !== a && null !== a.memoizedState !== d && (b.flags |= 8192), d && 0 !== (b.mode & 1) ? 0 !== (fj & 1073741824) && (S(b), b.subtreeFlags & 6 && (b.flags |= 8192)) : S(b), null;
    case 24:
      return null;
    case 25:
      return null;
  }
  throw Error(p(156, b.tag));
}
function Ij(a, b) {
  wg(b);
  switch (b.tag) {
    case 1:
      return Zf(b.type) && $f(), a = b.flags, a & 65536 ? (b.flags = a & -65537 | 128, b) : null;
    case 3:
      return zh(), E(Wf), E(H), Eh(), a = b.flags, 0 !== (a & 65536) && 0 === (a & 128) ? (b.flags = a & -65537 | 128, b) : null;
    case 5:
      return Bh(b), null;
    case 13:
      E(L);
      a = b.memoizedState;
      if (null !== a && null !== a.dehydrated) {
        if (null === b.alternate)
          throw Error(p(340));
        Ig();
      }
      a = b.flags;
      return a & 65536 ? (b.flags = a & -65537 | 128, b) : null;
    case 19:
      return E(L), null;
    case 4:
      return zh(), null;
    case 10:
      return ah(b.type._context), null;
    case 22:
    case 23:
      return Hj(), null;
    case 24:
      return null;
    default:
      return null;
  }
}
var Jj = false, U = false, Kj = "function" === typeof WeakSet ? WeakSet : Set, V = null;
function Lj(a, b) {
  var c = a.ref;
  if (null !== c)
    if ("function" === typeof c)
      try {
        c(null);
      } catch (d) {
        W(a, b, d);
      }
    else
      c.current = null;
}
function Mj(a, b, c) {
  try {
    c();
  } catch (d) {
    W(a, b, d);
  }
}
var Nj = false;
function Oj(a, b) {
  Cf = dd;
  a = Me();
  if (Ne(a)) {
    if ("selectionStart" in a)
      var c = { start: a.selectionStart, end: a.selectionEnd };
    else
      a: {
        c = (c = a.ownerDocument) && c.defaultView || window;
        var d = c.getSelection && c.getSelection();
        if (d && 0 !== d.rangeCount) {
          c = d.anchorNode;
          var e = d.anchorOffset, f2 = d.focusNode;
          d = d.focusOffset;
          try {
            c.nodeType, f2.nodeType;
          } catch (F2) {
            c = null;
            break a;
          }
          var g = 0, h = -1, k2 = -1, l2 = 0, m2 = 0, q2 = a, r2 = null;
          b:
            for (; ; ) {
              for (var y2; ; ) {
                q2 !== c || 0 !== e && 3 !== q2.nodeType || (h = g + e);
                q2 !== f2 || 0 !== d && 3 !== q2.nodeType || (k2 = g + d);
                3 === q2.nodeType && (g += q2.nodeValue.length);
                if (null === (y2 = q2.firstChild))
                  break;
                r2 = q2;
                q2 = y2;
              }
              for (; ; ) {
                if (q2 === a)
                  break b;
                r2 === c && ++l2 === e && (h = g);
                r2 === f2 && ++m2 === d && (k2 = g);
                if (null !== (y2 = q2.nextSibling))
                  break;
                q2 = r2;
                r2 = q2.parentNode;
              }
              q2 = y2;
            }
          c = -1 === h || -1 === k2 ? null : { start: h, end: k2 };
        } else
          c = null;
      }
    c = c || { start: 0, end: 0 };
  } else
    c = null;
  Df = { focusedElem: a, selectionRange: c };
  dd = false;
  for (V = b; null !== V; )
    if (b = V, a = b.child, 0 !== (b.subtreeFlags & 1028) && null !== a)
      a.return = b, V = a;
    else
      for (; null !== V; ) {
        b = V;
        try {
          var n2 = b.alternate;
          if (0 !== (b.flags & 1024))
            switch (b.tag) {
              case 0:
              case 11:
              case 15:
                break;
              case 1:
                if (null !== n2) {
                  var t2 = n2.memoizedProps, J2 = n2.memoizedState, x2 = b.stateNode, w2 = x2.getSnapshotBeforeUpdate(b.elementType === b.type ? t2 : Ci(b.type, t2), J2);
                  x2.__reactInternalSnapshotBeforeUpdate = w2;
                }
                break;
              case 3:
                var u2 = b.stateNode.containerInfo;
                1 === u2.nodeType ? u2.textContent = "" : 9 === u2.nodeType && u2.documentElement && u2.removeChild(u2.documentElement);
                break;
              case 5:
              case 6:
              case 4:
              case 17:
                break;
              default:
                throw Error(p(163));
            }
        } catch (F2) {
          W(b, b.return, F2);
        }
        a = b.sibling;
        if (null !== a) {
          a.return = b.return;
          V = a;
          break;
        }
        V = b.return;
      }
  n2 = Nj;
  Nj = false;
  return n2;
}
function Pj(a, b, c) {
  var d = b.updateQueue;
  d = null !== d ? d.lastEffect : null;
  if (null !== d) {
    var e = d = d.next;
    do {
      if ((e.tag & a) === a) {
        var f2 = e.destroy;
        e.destroy = void 0;
        void 0 !== f2 && Mj(b, c, f2);
      }
      e = e.next;
    } while (e !== d);
  }
}
function Qj(a, b) {
  b = b.updateQueue;
  b = null !== b ? b.lastEffect : null;
  if (null !== b) {
    var c = b = b.next;
    do {
      if ((c.tag & a) === a) {
        var d = c.create;
        c.destroy = d();
      }
      c = c.next;
    } while (c !== b);
  }
}
function Rj(a) {
  var b = a.ref;
  if (null !== b) {
    var c = a.stateNode;
    switch (a.tag) {
      case 5:
        a = c;
        break;
      default:
        a = c;
    }
    "function" === typeof b ? b(a) : b.current = a;
  }
}
function Sj(a) {
  var b = a.alternate;
  null !== b && (a.alternate = null, Sj(b));
  a.child = null;
  a.deletions = null;
  a.sibling = null;
  5 === a.tag && (b = a.stateNode, null !== b && (delete b[Of], delete b[Pf], delete b[of], delete b[Qf], delete b[Rf]));
  a.stateNode = null;
  a.return = null;
  a.dependencies = null;
  a.memoizedProps = null;
  a.memoizedState = null;
  a.pendingProps = null;
  a.stateNode = null;
  a.updateQueue = null;
}
function Tj(a) {
  return 5 === a.tag || 3 === a.tag || 4 === a.tag;
}
function Uj(a) {
  a:
    for (; ; ) {
      for (; null === a.sibling; ) {
        if (null === a.return || Tj(a.return))
          return null;
        a = a.return;
      }
      a.sibling.return = a.return;
      for (a = a.sibling; 5 !== a.tag && 6 !== a.tag && 18 !== a.tag; ) {
        if (a.flags & 2)
          continue a;
        if (null === a.child || 4 === a.tag)
          continue a;
        else
          a.child.return = a, a = a.child;
      }
      if (!(a.flags & 2))
        return a.stateNode;
    }
}
function Vj(a, b, c) {
  var d = a.tag;
  if (5 === d || 6 === d)
    a = a.stateNode, b ? 8 === c.nodeType ? c.parentNode.insertBefore(a, b) : c.insertBefore(a, b) : (8 === c.nodeType ? (b = c.parentNode, b.insertBefore(a, c)) : (b = c, b.appendChild(a)), c = c._reactRootContainer, null !== c && void 0 !== c || null !== b.onclick || (b.onclick = Bf));
  else if (4 !== d && (a = a.child, null !== a))
    for (Vj(a, b, c), a = a.sibling; null !== a; )
      Vj(a, b, c), a = a.sibling;
}
function Wj(a, b, c) {
  var d = a.tag;
  if (5 === d || 6 === d)
    a = a.stateNode, b ? c.insertBefore(a, b) : c.appendChild(a);
  else if (4 !== d && (a = a.child, null !== a))
    for (Wj(a, b, c), a = a.sibling; null !== a; )
      Wj(a, b, c), a = a.sibling;
}
var X$1 = null, Xj = false;
function Yj(a, b, c) {
  for (c = c.child; null !== c; )
    Zj(a, b, c), c = c.sibling;
}
function Zj(a, b, c) {
  if (lc && "function" === typeof lc.onCommitFiberUnmount)
    try {
      lc.onCommitFiberUnmount(kc, c);
    } catch (h) {
    }
  switch (c.tag) {
    case 5:
      U || Lj(c, b);
    case 6:
      var d = X$1, e = Xj;
      X$1 = null;
      Yj(a, b, c);
      X$1 = d;
      Xj = e;
      null !== X$1 && (Xj ? (a = X$1, c = c.stateNode, 8 === a.nodeType ? a.parentNode.removeChild(c) : a.removeChild(c)) : X$1.removeChild(c.stateNode));
      break;
    case 18:
      null !== X$1 && (Xj ? (a = X$1, c = c.stateNode, 8 === a.nodeType ? Kf(a.parentNode, c) : 1 === a.nodeType && Kf(a, c), bd(a)) : Kf(X$1, c.stateNode));
      break;
    case 4:
      d = X$1;
      e = Xj;
      X$1 = c.stateNode.containerInfo;
      Xj = true;
      Yj(a, b, c);
      X$1 = d;
      Xj = e;
      break;
    case 0:
    case 11:
    case 14:
    case 15:
      if (!U && (d = c.updateQueue, null !== d && (d = d.lastEffect, null !== d))) {
        e = d = d.next;
        do {
          var f2 = e, g = f2.destroy;
          f2 = f2.tag;
          void 0 !== g && (0 !== (f2 & 2) ? Mj(c, b, g) : 0 !== (f2 & 4) && Mj(c, b, g));
          e = e.next;
        } while (e !== d);
      }
      Yj(a, b, c);
      break;
    case 1:
      if (!U && (Lj(c, b), d = c.stateNode, "function" === typeof d.componentWillUnmount))
        try {
          d.props = c.memoizedProps, d.state = c.memoizedState, d.componentWillUnmount();
        } catch (h) {
          W(c, b, h);
        }
      Yj(a, b, c);
      break;
    case 21:
      Yj(a, b, c);
      break;
    case 22:
      c.mode & 1 ? (U = (d = U) || null !== c.memoizedState, Yj(a, b, c), U = d) : Yj(a, b, c);
      break;
    default:
      Yj(a, b, c);
  }
}
function ak(a) {
  var b = a.updateQueue;
  if (null !== b) {
    a.updateQueue = null;
    var c = a.stateNode;
    null === c && (c = a.stateNode = new Kj());
    b.forEach(function(b2) {
      var d = bk.bind(null, a, b2);
      c.has(b2) || (c.add(b2), b2.then(d, d));
    });
  }
}
function ck(a, b) {
  var c = b.deletions;
  if (null !== c)
    for (var d = 0; d < c.length; d++) {
      var e = c[d];
      try {
        var f2 = a, g = b, h = g;
        a:
          for (; null !== h; ) {
            switch (h.tag) {
              case 5:
                X$1 = h.stateNode;
                Xj = false;
                break a;
              case 3:
                X$1 = h.stateNode.containerInfo;
                Xj = true;
                break a;
              case 4:
                X$1 = h.stateNode.containerInfo;
                Xj = true;
                break a;
            }
            h = h.return;
          }
        if (null === X$1)
          throw Error(p(160));
        Zj(f2, g, e);
        X$1 = null;
        Xj = false;
        var k2 = e.alternate;
        null !== k2 && (k2.return = null);
        e.return = null;
      } catch (l2) {
        W(e, b, l2);
      }
    }
  if (b.subtreeFlags & 12854)
    for (b = b.child; null !== b; )
      dk(b, a), b = b.sibling;
}
function dk(a, b) {
  var c = a.alternate, d = a.flags;
  switch (a.tag) {
    case 0:
    case 11:
    case 14:
    case 15:
      ck(b, a);
      ek(a);
      if (d & 4) {
        try {
          Pj(3, a, a.return), Qj(3, a);
        } catch (t2) {
          W(a, a.return, t2);
        }
        try {
          Pj(5, a, a.return);
        } catch (t2) {
          W(a, a.return, t2);
        }
      }
      break;
    case 1:
      ck(b, a);
      ek(a);
      d & 512 && null !== c && Lj(c, c.return);
      break;
    case 5:
      ck(b, a);
      ek(a);
      d & 512 && null !== c && Lj(c, c.return);
      if (a.flags & 32) {
        var e = a.stateNode;
        try {
          ob(e, "");
        } catch (t2) {
          W(a, a.return, t2);
        }
      }
      if (d & 4 && (e = a.stateNode, null != e)) {
        var f2 = a.memoizedProps, g = null !== c ? c.memoizedProps : f2, h = a.type, k2 = a.updateQueue;
        a.updateQueue = null;
        if (null !== k2)
          try {
            "input" === h && "radio" === f2.type && null != f2.name && ab(e, f2);
            vb(h, g);
            var l2 = vb(h, f2);
            for (g = 0; g < k2.length; g += 2) {
              var m2 = k2[g], q2 = k2[g + 1];
              "style" === m2 ? sb(e, q2) : "dangerouslySetInnerHTML" === m2 ? nb(e, q2) : "children" === m2 ? ob(e, q2) : ta(e, m2, q2, l2);
            }
            switch (h) {
              case "input":
                bb(e, f2);
                break;
              case "textarea":
                ib(e, f2);
                break;
              case "select":
                var r2 = e._wrapperState.wasMultiple;
                e._wrapperState.wasMultiple = !!f2.multiple;
                var y2 = f2.value;
                null != y2 ? fb(e, !!f2.multiple, y2, false) : r2 !== !!f2.multiple && (null != f2.defaultValue ? fb(
                  e,
                  !!f2.multiple,
                  f2.defaultValue,
                  true
                ) : fb(e, !!f2.multiple, f2.multiple ? [] : "", false));
            }
            e[Pf] = f2;
          } catch (t2) {
            W(a, a.return, t2);
          }
      }
      break;
    case 6:
      ck(b, a);
      ek(a);
      if (d & 4) {
        if (null === a.stateNode)
          throw Error(p(162));
        e = a.stateNode;
        f2 = a.memoizedProps;
        try {
          e.nodeValue = f2;
        } catch (t2) {
          W(a, a.return, t2);
        }
      }
      break;
    case 3:
      ck(b, a);
      ek(a);
      if (d & 4 && null !== c && c.memoizedState.isDehydrated)
        try {
          bd(b.containerInfo);
        } catch (t2) {
          W(a, a.return, t2);
        }
      break;
    case 4:
      ck(b, a);
      ek(a);
      break;
    case 13:
      ck(b, a);
      ek(a);
      e = a.child;
      e.flags & 8192 && (f2 = null !== e.memoizedState, e.stateNode.isHidden = f2, !f2 || null !== e.alternate && null !== e.alternate.memoizedState || (fk = B()));
      d & 4 && ak(a);
      break;
    case 22:
      m2 = null !== c && null !== c.memoizedState;
      a.mode & 1 ? (U = (l2 = U) || m2, ck(b, a), U = l2) : ck(b, a);
      ek(a);
      if (d & 8192) {
        l2 = null !== a.memoizedState;
        if ((a.stateNode.isHidden = l2) && !m2 && 0 !== (a.mode & 1))
          for (V = a, m2 = a.child; null !== m2; ) {
            for (q2 = V = m2; null !== V; ) {
              r2 = V;
              y2 = r2.child;
              switch (r2.tag) {
                case 0:
                case 11:
                case 14:
                case 15:
                  Pj(4, r2, r2.return);
                  break;
                case 1:
                  Lj(r2, r2.return);
                  var n2 = r2.stateNode;
                  if ("function" === typeof n2.componentWillUnmount) {
                    d = r2;
                    c = r2.return;
                    try {
                      b = d, n2.props = b.memoizedProps, n2.state = b.memoizedState, n2.componentWillUnmount();
                    } catch (t2) {
                      W(d, c, t2);
                    }
                  }
                  break;
                case 5:
                  Lj(r2, r2.return);
                  break;
                case 22:
                  if (null !== r2.memoizedState) {
                    gk(q2);
                    continue;
                  }
              }
              null !== y2 ? (y2.return = r2, V = y2) : gk(q2);
            }
            m2 = m2.sibling;
          }
        a:
          for (m2 = null, q2 = a; ; ) {
            if (5 === q2.tag) {
              if (null === m2) {
                m2 = q2;
                try {
                  e = q2.stateNode, l2 ? (f2 = e.style, "function" === typeof f2.setProperty ? f2.setProperty("display", "none", "important") : f2.display = "none") : (h = q2.stateNode, k2 = q2.memoizedProps.style, g = void 0 !== k2 && null !== k2 && k2.hasOwnProperty("display") ? k2.display : null, h.style.display = rb("display", g));
                } catch (t2) {
                  W(a, a.return, t2);
                }
              }
            } else if (6 === q2.tag) {
              if (null === m2)
                try {
                  q2.stateNode.nodeValue = l2 ? "" : q2.memoizedProps;
                } catch (t2) {
                  W(a, a.return, t2);
                }
            } else if ((22 !== q2.tag && 23 !== q2.tag || null === q2.memoizedState || q2 === a) && null !== q2.child) {
              q2.child.return = q2;
              q2 = q2.child;
              continue;
            }
            if (q2 === a)
              break a;
            for (; null === q2.sibling; ) {
              if (null === q2.return || q2.return === a)
                break a;
              m2 === q2 && (m2 = null);
              q2 = q2.return;
            }
            m2 === q2 && (m2 = null);
            q2.sibling.return = q2.return;
            q2 = q2.sibling;
          }
      }
      break;
    case 19:
      ck(b, a);
      ek(a);
      d & 4 && ak(a);
      break;
    case 21:
      break;
    default:
      ck(
        b,
        a
      ), ek(a);
  }
}
function ek(a) {
  var b = a.flags;
  if (b & 2) {
    try {
      a: {
        for (var c = a.return; null !== c; ) {
          if (Tj(c)) {
            var d = c;
            break a;
          }
          c = c.return;
        }
        throw Error(p(160));
      }
      switch (d.tag) {
        case 5:
          var e = d.stateNode;
          d.flags & 32 && (ob(e, ""), d.flags &= -33);
          var f2 = Uj(a);
          Wj(a, f2, e);
          break;
        case 3:
        case 4:
          var g = d.stateNode.containerInfo, h = Uj(a);
          Vj(a, h, g);
          break;
        default:
          throw Error(p(161));
      }
    } catch (k2) {
      W(a, a.return, k2);
    }
    a.flags &= -3;
  }
  b & 4096 && (a.flags &= -4097);
}
function hk(a, b, c) {
  V = a;
  ik(a);
}
function ik(a, b, c) {
  for (var d = 0 !== (a.mode & 1); null !== V; ) {
    var e = V, f2 = e.child;
    if (22 === e.tag && d) {
      var g = null !== e.memoizedState || Jj;
      if (!g) {
        var h = e.alternate, k2 = null !== h && null !== h.memoizedState || U;
        h = Jj;
        var l2 = U;
        Jj = g;
        if ((U = k2) && !l2)
          for (V = e; null !== V; )
            g = V, k2 = g.child, 22 === g.tag && null !== g.memoizedState ? jk(e) : null !== k2 ? (k2.return = g, V = k2) : jk(e);
        for (; null !== f2; )
          V = f2, ik(f2), f2 = f2.sibling;
        V = e;
        Jj = h;
        U = l2;
      }
      kk(a);
    } else
      0 !== (e.subtreeFlags & 8772) && null !== f2 ? (f2.return = e, V = f2) : kk(a);
  }
}
function kk(a) {
  for (; null !== V; ) {
    var b = V;
    if (0 !== (b.flags & 8772)) {
      var c = b.alternate;
      try {
        if (0 !== (b.flags & 8772))
          switch (b.tag) {
            case 0:
            case 11:
            case 15:
              U || Qj(5, b);
              break;
            case 1:
              var d = b.stateNode;
              if (b.flags & 4 && !U)
                if (null === c)
                  d.componentDidMount();
                else {
                  var e = b.elementType === b.type ? c.memoizedProps : Ci(b.type, c.memoizedProps);
                  d.componentDidUpdate(e, c.memoizedState, d.__reactInternalSnapshotBeforeUpdate);
                }
              var f2 = b.updateQueue;
              null !== f2 && sh(b, f2, d);
              break;
            case 3:
              var g = b.updateQueue;
              if (null !== g) {
                c = null;
                if (null !== b.child)
                  switch (b.child.tag) {
                    case 5:
                      c = b.child.stateNode;
                      break;
                    case 1:
                      c = b.child.stateNode;
                  }
                sh(b, g, c);
              }
              break;
            case 5:
              var h = b.stateNode;
              if (null === c && b.flags & 4) {
                c = h;
                var k2 = b.memoizedProps;
                switch (b.type) {
                  case "button":
                  case "input":
                  case "select":
                  case "textarea":
                    k2.autoFocus && c.focus();
                    break;
                  case "img":
                    k2.src && (c.src = k2.src);
                }
              }
              break;
            case 6:
              break;
            case 4:
              break;
            case 12:
              break;
            case 13:
              if (null === b.memoizedState) {
                var l2 = b.alternate;
                if (null !== l2) {
                  var m2 = l2.memoizedState;
                  if (null !== m2) {
                    var q2 = m2.dehydrated;
                    null !== q2 && bd(q2);
                  }
                }
              }
              break;
            case 19:
            case 17:
            case 21:
            case 22:
            case 23:
            case 25:
              break;
            default:
              throw Error(p(163));
          }
        U || b.flags & 512 && Rj(b);
      } catch (r2) {
        W(b, b.return, r2);
      }
    }
    if (b === a) {
      V = null;
      break;
    }
    c = b.sibling;
    if (null !== c) {
      c.return = b.return;
      V = c;
      break;
    }
    V = b.return;
  }
}
function gk(a) {
  for (; null !== V; ) {
    var b = V;
    if (b === a) {
      V = null;
      break;
    }
    var c = b.sibling;
    if (null !== c) {
      c.return = b.return;
      V = c;
      break;
    }
    V = b.return;
  }
}
function jk(a) {
  for (; null !== V; ) {
    var b = V;
    try {
      switch (b.tag) {
        case 0:
        case 11:
        case 15:
          var c = b.return;
          try {
            Qj(4, b);
          } catch (k2) {
            W(b, c, k2);
          }
          break;
        case 1:
          var d = b.stateNode;
          if ("function" === typeof d.componentDidMount) {
            var e = b.return;
            try {
              d.componentDidMount();
            } catch (k2) {
              W(b, e, k2);
            }
          }
          var f2 = b.return;
          try {
            Rj(b);
          } catch (k2) {
            W(b, f2, k2);
          }
          break;
        case 5:
          var g = b.return;
          try {
            Rj(b);
          } catch (k2) {
            W(b, g, k2);
          }
      }
    } catch (k2) {
      W(b, b.return, k2);
    }
    if (b === a) {
      V = null;
      break;
    }
    var h = b.sibling;
    if (null !== h) {
      h.return = b.return;
      V = h;
      break;
    }
    V = b.return;
  }
}
var lk = Math.ceil, mk = ua.ReactCurrentDispatcher, nk = ua.ReactCurrentOwner, ok = ua.ReactCurrentBatchConfig, K = 0, Q = null, Y = null, Z = 0, fj = 0, ej = Uf(0), T = 0, pk = null, rh = 0, qk = 0, rk = 0, sk = null, tk = null, fk = 0, Gj = Infinity, uk = null, Oi = false, Pi = null, Ri = null, vk = false, wk = null, xk = 0, yk = 0, zk = null, Ak = -1, Bk = 0;
function R() {
  return 0 !== (K & 6) ? B() : -1 !== Ak ? Ak : Ak = B();
}
function yi(a) {
  if (0 === (a.mode & 1))
    return 1;
  if (0 !== (K & 2) && 0 !== Z)
    return Z & -Z;
  if (null !== Kg.transition)
    return 0 === Bk && (Bk = yc()), Bk;
  a = C;
  if (0 !== a)
    return a;
  a = window.event;
  a = void 0 === a ? 16 : jd(a.type);
  return a;
}
function gi(a, b, c, d) {
  if (50 < yk)
    throw yk = 0, zk = null, Error(p(185));
  Ac(a, c, d);
  if (0 === (K & 2) || a !== Q)
    a === Q && (0 === (K & 2) && (qk |= c), 4 === T && Ck(a, Z)), Dk(a, d), 1 === c && 0 === K && 0 === (b.mode & 1) && (Gj = B() + 500, fg && jg());
}
function Dk(a, b) {
  var c = a.callbackNode;
  wc(a, b);
  var d = uc(a, a === Q ? Z : 0);
  if (0 === d)
    null !== c && bc(c), a.callbackNode = null, a.callbackPriority = 0;
  else if (b = d & -d, a.callbackPriority !== b) {
    null != c && bc(c);
    if (1 === b)
      0 === a.tag ? ig(Ek.bind(null, a)) : hg(Ek.bind(null, a)), Jf(function() {
        0 === (K & 6) && jg();
      }), c = null;
    else {
      switch (Dc(d)) {
        case 1:
          c = fc;
          break;
        case 4:
          c = gc;
          break;
        case 16:
          c = hc;
          break;
        case 536870912:
          c = jc;
          break;
        default:
          c = hc;
      }
      c = Fk(c, Gk.bind(null, a));
    }
    a.callbackPriority = b;
    a.callbackNode = c;
  }
}
function Gk(a, b) {
  Ak = -1;
  Bk = 0;
  if (0 !== (K & 6))
    throw Error(p(327));
  var c = a.callbackNode;
  if (Hk() && a.callbackNode !== c)
    return null;
  var d = uc(a, a === Q ? Z : 0);
  if (0 === d)
    return null;
  if (0 !== (d & 30) || 0 !== (d & a.expiredLanes) || b)
    b = Ik(a, d);
  else {
    b = d;
    var e = K;
    K |= 2;
    var f2 = Jk();
    if (Q !== a || Z !== b)
      uk = null, Gj = B() + 500, Kk(a, b);
    do
      try {
        Lk();
        break;
      } catch (h) {
        Mk(a, h);
      }
    while (1);
    $g();
    mk.current = f2;
    K = e;
    null !== Y ? b = 0 : (Q = null, Z = 0, b = T);
  }
  if (0 !== b) {
    2 === b && (e = xc(a), 0 !== e && (d = e, b = Nk(a, e)));
    if (1 === b)
      throw c = pk, Kk(a, 0), Ck(a, d), Dk(a, B()), c;
    if (6 === b)
      Ck(a, d);
    else {
      e = a.current.alternate;
      if (0 === (d & 30) && !Ok(e) && (b = Ik(a, d), 2 === b && (f2 = xc(a), 0 !== f2 && (d = f2, b = Nk(a, f2))), 1 === b))
        throw c = pk, Kk(a, 0), Ck(a, d), Dk(a, B()), c;
      a.finishedWork = e;
      a.finishedLanes = d;
      switch (b) {
        case 0:
        case 1:
          throw Error(p(345));
        case 2:
          Pk(a, tk, uk);
          break;
        case 3:
          Ck(a, d);
          if ((d & 130023424) === d && (b = fk + 500 - B(), 10 < b)) {
            if (0 !== uc(a, 0))
              break;
            e = a.suspendedLanes;
            if ((e & d) !== d) {
              R();
              a.pingedLanes |= a.suspendedLanes & e;
              break;
            }
            a.timeoutHandle = Ff(Pk.bind(null, a, tk, uk), b);
            break;
          }
          Pk(a, tk, uk);
          break;
        case 4:
          Ck(a, d);
          if ((d & 4194240) === d)
            break;
          b = a.eventTimes;
          for (e = -1; 0 < d; ) {
            var g = 31 - oc(d);
            f2 = 1 << g;
            g = b[g];
            g > e && (e = g);
            d &= ~f2;
          }
          d = e;
          d = B() - d;
          d = (120 > d ? 120 : 480 > d ? 480 : 1080 > d ? 1080 : 1920 > d ? 1920 : 3e3 > d ? 3e3 : 4320 > d ? 4320 : 1960 * lk(d / 1960)) - d;
          if (10 < d) {
            a.timeoutHandle = Ff(Pk.bind(null, a, tk, uk), d);
            break;
          }
          Pk(a, tk, uk);
          break;
        case 5:
          Pk(a, tk, uk);
          break;
        default:
          throw Error(p(329));
      }
    }
  }
  Dk(a, B());
  return a.callbackNode === c ? Gk.bind(null, a) : null;
}
function Nk(a, b) {
  var c = sk;
  a.current.memoizedState.isDehydrated && (Kk(a, b).flags |= 256);
  a = Ik(a, b);
  2 !== a && (b = tk, tk = c, null !== b && Fj(b));
  return a;
}
function Fj(a) {
  null === tk ? tk = a : tk.push.apply(tk, a);
}
function Ok(a) {
  for (var b = a; ; ) {
    if (b.flags & 16384) {
      var c = b.updateQueue;
      if (null !== c && (c = c.stores, null !== c))
        for (var d = 0; d < c.length; d++) {
          var e = c[d], f2 = e.getSnapshot;
          e = e.value;
          try {
            if (!He(f2(), e))
              return false;
          } catch (g) {
            return false;
          }
        }
    }
    c = b.child;
    if (b.subtreeFlags & 16384 && null !== c)
      c.return = b, b = c;
    else {
      if (b === a)
        break;
      for (; null === b.sibling; ) {
        if (null === b.return || b.return === a)
          return true;
        b = b.return;
      }
      b.sibling.return = b.return;
      b = b.sibling;
    }
  }
  return true;
}
function Ck(a, b) {
  b &= ~rk;
  b &= ~qk;
  a.suspendedLanes |= b;
  a.pingedLanes &= ~b;
  for (a = a.expirationTimes; 0 < b; ) {
    var c = 31 - oc(b), d = 1 << c;
    a[c] = -1;
    b &= ~d;
  }
}
function Ek(a) {
  if (0 !== (K & 6))
    throw Error(p(327));
  Hk();
  var b = uc(a, 0);
  if (0 === (b & 1))
    return Dk(a, B()), null;
  var c = Ik(a, b);
  if (0 !== a.tag && 2 === c) {
    var d = xc(a);
    0 !== d && (b = d, c = Nk(a, d));
  }
  if (1 === c)
    throw c = pk, Kk(a, 0), Ck(a, b), Dk(a, B()), c;
  if (6 === c)
    throw Error(p(345));
  a.finishedWork = a.current.alternate;
  a.finishedLanes = b;
  Pk(a, tk, uk);
  Dk(a, B());
  return null;
}
function Qk(a, b) {
  var c = K;
  K |= 1;
  try {
    return a(b);
  } finally {
    K = c, 0 === K && (Gj = B() + 500, fg && jg());
  }
}
function Rk(a) {
  null !== wk && 0 === wk.tag && 0 === (K & 6) && Hk();
  var b = K;
  K |= 1;
  var c = ok.transition, d = C;
  try {
    if (ok.transition = null, C = 1, a)
      return a();
  } finally {
    C = d, ok.transition = c, K = b, 0 === (K & 6) && jg();
  }
}
function Hj() {
  fj = ej.current;
  E(ej);
}
function Kk(a, b) {
  a.finishedWork = null;
  a.finishedLanes = 0;
  var c = a.timeoutHandle;
  -1 !== c && (a.timeoutHandle = -1, Gf(c));
  if (null !== Y)
    for (c = Y.return; null !== c; ) {
      var d = c;
      wg(d);
      switch (d.tag) {
        case 1:
          d = d.type.childContextTypes;
          null !== d && void 0 !== d && $f();
          break;
        case 3:
          zh();
          E(Wf);
          E(H);
          Eh();
          break;
        case 5:
          Bh(d);
          break;
        case 4:
          zh();
          break;
        case 13:
          E(L);
          break;
        case 19:
          E(L);
          break;
        case 10:
          ah(d.type._context);
          break;
        case 22:
        case 23:
          Hj();
      }
      c = c.return;
    }
  Q = a;
  Y = a = Pg(a.current, null);
  Z = fj = b;
  T = 0;
  pk = null;
  rk = qk = rh = 0;
  tk = sk = null;
  if (null !== fh) {
    for (b = 0; b < fh.length; b++)
      if (c = fh[b], d = c.interleaved, null !== d) {
        c.interleaved = null;
        var e = d.next, f2 = c.pending;
        if (null !== f2) {
          var g = f2.next;
          f2.next = e;
          d.next = g;
        }
        c.pending = d;
      }
    fh = null;
  }
  return a;
}
function Mk(a, b) {
  do {
    var c = Y;
    try {
      $g();
      Fh.current = Rh;
      if (Ih) {
        for (var d = M.memoizedState; null !== d; ) {
          var e = d.queue;
          null !== e && (e.pending = null);
          d = d.next;
        }
        Ih = false;
      }
      Hh = 0;
      O = N = M = null;
      Jh = false;
      Kh = 0;
      nk.current = null;
      if (null === c || null === c.return) {
        T = 1;
        pk = b;
        Y = null;
        break;
      }
      a: {
        var f2 = a, g = c.return, h = c, k2 = b;
        b = Z;
        h.flags |= 32768;
        if (null !== k2 && "object" === typeof k2 && "function" === typeof k2.then) {
          var l2 = k2, m2 = h, q2 = m2.tag;
          if (0 === (m2.mode & 1) && (0 === q2 || 11 === q2 || 15 === q2)) {
            var r2 = m2.alternate;
            r2 ? (m2.updateQueue = r2.updateQueue, m2.memoizedState = r2.memoizedState, m2.lanes = r2.lanes) : (m2.updateQueue = null, m2.memoizedState = null);
          }
          var y2 = Ui(g);
          if (null !== y2) {
            y2.flags &= -257;
            Vi(y2, g, h, f2, b);
            y2.mode & 1 && Si(f2, l2, b);
            b = y2;
            k2 = l2;
            var n2 = b.updateQueue;
            if (null === n2) {
              var t2 = /* @__PURE__ */ new Set();
              t2.add(k2);
              b.updateQueue = t2;
            } else
              n2.add(k2);
            break a;
          } else {
            if (0 === (b & 1)) {
              Si(f2, l2, b);
              tj();
              break a;
            }
            k2 = Error(p(426));
          }
        } else if (I && h.mode & 1) {
          var J2 = Ui(g);
          if (null !== J2) {
            0 === (J2.flags & 65536) && (J2.flags |= 256);
            Vi(J2, g, h, f2, b);
            Jg(Ji(k2, h));
            break a;
          }
        }
        f2 = k2 = Ji(k2, h);
        4 !== T && (T = 2);
        null === sk ? sk = [f2] : sk.push(f2);
        f2 = g;
        do {
          switch (f2.tag) {
            case 3:
              f2.flags |= 65536;
              b &= -b;
              f2.lanes |= b;
              var x2 = Ni(f2, k2, b);
              ph(f2, x2);
              break a;
            case 1:
              h = k2;
              var w2 = f2.type, u2 = f2.stateNode;
              if (0 === (f2.flags & 128) && ("function" === typeof w2.getDerivedStateFromError || null !== u2 && "function" === typeof u2.componentDidCatch && (null === Ri || !Ri.has(u2)))) {
                f2.flags |= 65536;
                b &= -b;
                f2.lanes |= b;
                var F2 = Qi(f2, h, b);
                ph(f2, F2);
                break a;
              }
          }
          f2 = f2.return;
        } while (null !== f2);
      }
      Sk(c);
    } catch (na) {
      b = na;
      Y === c && null !== c && (Y = c = c.return);
      continue;
    }
    break;
  } while (1);
}
function Jk() {
  var a = mk.current;
  mk.current = Rh;
  return null === a ? Rh : a;
}
function tj() {
  if (0 === T || 3 === T || 2 === T)
    T = 4;
  null === Q || 0 === (rh & 268435455) && 0 === (qk & 268435455) || Ck(Q, Z);
}
function Ik(a, b) {
  var c = K;
  K |= 2;
  var d = Jk();
  if (Q !== a || Z !== b)
    uk = null, Kk(a, b);
  do
    try {
      Tk();
      break;
    } catch (e) {
      Mk(a, e);
    }
  while (1);
  $g();
  K = c;
  mk.current = d;
  if (null !== Y)
    throw Error(p(261));
  Q = null;
  Z = 0;
  return T;
}
function Tk() {
  for (; null !== Y; )
    Uk(Y);
}
function Lk() {
  for (; null !== Y && !cc(); )
    Uk(Y);
}
function Uk(a) {
  var b = Vk(a.alternate, a, fj);
  a.memoizedProps = a.pendingProps;
  null === b ? Sk(a) : Y = b;
  nk.current = null;
}
function Sk(a) {
  var b = a;
  do {
    var c = b.alternate;
    a = b.return;
    if (0 === (b.flags & 32768)) {
      if (c = Ej(c, b, fj), null !== c) {
        Y = c;
        return;
      }
    } else {
      c = Ij(c, b);
      if (null !== c) {
        c.flags &= 32767;
        Y = c;
        return;
      }
      if (null !== a)
        a.flags |= 32768, a.subtreeFlags = 0, a.deletions = null;
      else {
        T = 6;
        Y = null;
        return;
      }
    }
    b = b.sibling;
    if (null !== b) {
      Y = b;
      return;
    }
    Y = b = a;
  } while (null !== b);
  0 === T && (T = 5);
}
function Pk(a, b, c) {
  var d = C, e = ok.transition;
  try {
    ok.transition = null, C = 1, Wk(a, b, c, d);
  } finally {
    ok.transition = e, C = d;
  }
  return null;
}
function Wk(a, b, c, d) {
  do
    Hk();
  while (null !== wk);
  if (0 !== (K & 6))
    throw Error(p(327));
  c = a.finishedWork;
  var e = a.finishedLanes;
  if (null === c)
    return null;
  a.finishedWork = null;
  a.finishedLanes = 0;
  if (c === a.current)
    throw Error(p(177));
  a.callbackNode = null;
  a.callbackPriority = 0;
  var f2 = c.lanes | c.childLanes;
  Bc(a, f2);
  a === Q && (Y = Q = null, Z = 0);
  0 === (c.subtreeFlags & 2064) && 0 === (c.flags & 2064) || vk || (vk = true, Fk(hc, function() {
    Hk();
    return null;
  }));
  f2 = 0 !== (c.flags & 15990);
  if (0 !== (c.subtreeFlags & 15990) || f2) {
    f2 = ok.transition;
    ok.transition = null;
    var g = C;
    C = 1;
    var h = K;
    K |= 4;
    nk.current = null;
    Oj(a, c);
    dk(c, a);
    Oe(Df);
    dd = !!Cf;
    Df = Cf = null;
    a.current = c;
    hk(c);
    dc();
    K = h;
    C = g;
    ok.transition = f2;
  } else
    a.current = c;
  vk && (vk = false, wk = a, xk = e);
  f2 = a.pendingLanes;
  0 === f2 && (Ri = null);
  mc(c.stateNode);
  Dk(a, B());
  if (null !== b)
    for (d = a.onRecoverableError, c = 0; c < b.length; c++)
      e = b[c], d(e.value, { componentStack: e.stack, digest: e.digest });
  if (Oi)
    throw Oi = false, a = Pi, Pi = null, a;
  0 !== (xk & 1) && 0 !== a.tag && Hk();
  f2 = a.pendingLanes;
  0 !== (f2 & 1) ? a === zk ? yk++ : (yk = 0, zk = a) : yk = 0;
  jg();
  return null;
}
function Hk() {
  if (null !== wk) {
    var a = Dc(xk), b = ok.transition, c = C;
    try {
      ok.transition = null;
      C = 16 > a ? 16 : a;
      if (null === wk)
        var d = false;
      else {
        a = wk;
        wk = null;
        xk = 0;
        if (0 !== (K & 6))
          throw Error(p(331));
        var e = K;
        K |= 4;
        for (V = a.current; null !== V; ) {
          var f2 = V, g = f2.child;
          if (0 !== (V.flags & 16)) {
            var h = f2.deletions;
            if (null !== h) {
              for (var k2 = 0; k2 < h.length; k2++) {
                var l2 = h[k2];
                for (V = l2; null !== V; ) {
                  var m2 = V;
                  switch (m2.tag) {
                    case 0:
                    case 11:
                    case 15:
                      Pj(8, m2, f2);
                  }
                  var q2 = m2.child;
                  if (null !== q2)
                    q2.return = m2, V = q2;
                  else
                    for (; null !== V; ) {
                      m2 = V;
                      var r2 = m2.sibling, y2 = m2.return;
                      Sj(m2);
                      if (m2 === l2) {
                        V = null;
                        break;
                      }
                      if (null !== r2) {
                        r2.return = y2;
                        V = r2;
                        break;
                      }
                      V = y2;
                    }
                }
              }
              var n2 = f2.alternate;
              if (null !== n2) {
                var t2 = n2.child;
                if (null !== t2) {
                  n2.child = null;
                  do {
                    var J2 = t2.sibling;
                    t2.sibling = null;
                    t2 = J2;
                  } while (null !== t2);
                }
              }
              V = f2;
            }
          }
          if (0 !== (f2.subtreeFlags & 2064) && null !== g)
            g.return = f2, V = g;
          else
            b:
              for (; null !== V; ) {
                f2 = V;
                if (0 !== (f2.flags & 2048))
                  switch (f2.tag) {
                    case 0:
                    case 11:
                    case 15:
                      Pj(9, f2, f2.return);
                  }
                var x2 = f2.sibling;
                if (null !== x2) {
                  x2.return = f2.return;
                  V = x2;
                  break b;
                }
                V = f2.return;
              }
        }
        var w2 = a.current;
        for (V = w2; null !== V; ) {
          g = V;
          var u2 = g.child;
          if (0 !== (g.subtreeFlags & 2064) && null !== u2)
            u2.return = g, V = u2;
          else
            b:
              for (g = w2; null !== V; ) {
                h = V;
                if (0 !== (h.flags & 2048))
                  try {
                    switch (h.tag) {
                      case 0:
                      case 11:
                      case 15:
                        Qj(9, h);
                    }
                  } catch (na) {
                    W(h, h.return, na);
                  }
                if (h === g) {
                  V = null;
                  break b;
                }
                var F2 = h.sibling;
                if (null !== F2) {
                  F2.return = h.return;
                  V = F2;
                  break b;
                }
                V = h.return;
              }
        }
        K = e;
        jg();
        if (lc && "function" === typeof lc.onPostCommitFiberRoot)
          try {
            lc.onPostCommitFiberRoot(kc, a);
          } catch (na) {
          }
        d = true;
      }
      return d;
    } finally {
      C = c, ok.transition = b;
    }
  }
  return false;
}
function Xk(a, b, c) {
  b = Ji(c, b);
  b = Ni(a, b, 1);
  a = nh(a, b, 1);
  b = R();
  null !== a && (Ac(a, 1, b), Dk(a, b));
}
function W(a, b, c) {
  if (3 === a.tag)
    Xk(a, a, c);
  else
    for (; null !== b; ) {
      if (3 === b.tag) {
        Xk(b, a, c);
        break;
      } else if (1 === b.tag) {
        var d = b.stateNode;
        if ("function" === typeof b.type.getDerivedStateFromError || "function" === typeof d.componentDidCatch && (null === Ri || !Ri.has(d))) {
          a = Ji(c, a);
          a = Qi(b, a, 1);
          b = nh(b, a, 1);
          a = R();
          null !== b && (Ac(b, 1, a), Dk(b, a));
          break;
        }
      }
      b = b.return;
    }
}
function Ti(a, b, c) {
  var d = a.pingCache;
  null !== d && d.delete(b);
  b = R();
  a.pingedLanes |= a.suspendedLanes & c;
  Q === a && (Z & c) === c && (4 === T || 3 === T && (Z & 130023424) === Z && 500 > B() - fk ? Kk(a, 0) : rk |= c);
  Dk(a, b);
}
function Yk(a, b) {
  0 === b && (0 === (a.mode & 1) ? b = 1 : (b = sc, sc <<= 1, 0 === (sc & 130023424) && (sc = 4194304)));
  var c = R();
  a = ih(a, b);
  null !== a && (Ac(a, b, c), Dk(a, c));
}
function uj(a) {
  var b = a.memoizedState, c = 0;
  null !== b && (c = b.retryLane);
  Yk(a, c);
}
function bk(a, b) {
  var c = 0;
  switch (a.tag) {
    case 13:
      var d = a.stateNode;
      var e = a.memoizedState;
      null !== e && (c = e.retryLane);
      break;
    case 19:
      d = a.stateNode;
      break;
    default:
      throw Error(p(314));
  }
  null !== d && d.delete(b);
  Yk(a, c);
}
var Vk;
Vk = function(a, b, c) {
  if (null !== a)
    if (a.memoizedProps !== b.pendingProps || Wf.current)
      dh = true;
    else {
      if (0 === (a.lanes & c) && 0 === (b.flags & 128))
        return dh = false, yj(a, b, c);
      dh = 0 !== (a.flags & 131072) ? true : false;
    }
  else
    dh = false, I && 0 !== (b.flags & 1048576) && ug(b, ng, b.index);
  b.lanes = 0;
  switch (b.tag) {
    case 2:
      var d = b.type;
      ij(a, b);
      a = b.pendingProps;
      var e = Yf(b, H.current);
      ch(b, c);
      e = Nh(null, b, d, a, e, c);
      var f2 = Sh();
      b.flags |= 1;
      "object" === typeof e && null !== e && "function" === typeof e.render && void 0 === e.$$typeof ? (b.tag = 1, b.memoizedState = null, b.updateQueue = null, Zf(d) ? (f2 = true, cg(b)) : f2 = false, b.memoizedState = null !== e.state && void 0 !== e.state ? e.state : null, kh(b), e.updater = Ei, b.stateNode = e, e._reactInternals = b, Ii(b, d, a, c), b = jj(null, b, d, true, f2, c)) : (b.tag = 0, I && f2 && vg(b), Xi(null, b, e, c), b = b.child);
      return b;
    case 16:
      d = b.elementType;
      a: {
        ij(a, b);
        a = b.pendingProps;
        e = d._init;
        d = e(d._payload);
        b.type = d;
        e = b.tag = Zk(d);
        a = Ci(d, a);
        switch (e) {
          case 0:
            b = cj(null, b, d, a, c);
            break a;
          case 1:
            b = hj(null, b, d, a, c);
            break a;
          case 11:
            b = Yi(null, b, d, a, c);
            break a;
          case 14:
            b = $i(null, b, d, Ci(d.type, a), c);
            break a;
        }
        throw Error(p(
          306,
          d,
          ""
        ));
      }
      return b;
    case 0:
      return d = b.type, e = b.pendingProps, e = b.elementType === d ? e : Ci(d, e), cj(a, b, d, e, c);
    case 1:
      return d = b.type, e = b.pendingProps, e = b.elementType === d ? e : Ci(d, e), hj(a, b, d, e, c);
    case 3:
      a: {
        kj(b);
        if (null === a)
          throw Error(p(387));
        d = b.pendingProps;
        f2 = b.memoizedState;
        e = f2.element;
        lh(a, b);
        qh(b, d, null, c);
        var g = b.memoizedState;
        d = g.element;
        if (f2.isDehydrated)
          if (f2 = { element: d, isDehydrated: false, cache: g.cache, pendingSuspenseBoundaries: g.pendingSuspenseBoundaries, transitions: g.transitions }, b.updateQueue.baseState = f2, b.memoizedState = f2, b.flags & 256) {
            e = Ji(Error(p(423)), b);
            b = lj(a, b, d, c, e);
            break a;
          } else if (d !== e) {
            e = Ji(Error(p(424)), b);
            b = lj(a, b, d, c, e);
            break a;
          } else
            for (yg = Lf(b.stateNode.containerInfo.firstChild), xg = b, I = true, zg = null, c = Vg(b, null, d, c), b.child = c; c; )
              c.flags = c.flags & -3 | 4096, c = c.sibling;
        else {
          Ig();
          if (d === e) {
            b = Zi(a, b, c);
            break a;
          }
          Xi(a, b, d, c);
        }
        b = b.child;
      }
      return b;
    case 5:
      return Ah(b), null === a && Eg(b), d = b.type, e = b.pendingProps, f2 = null !== a ? a.memoizedProps : null, g = e.children, Ef(d, e) ? g = null : null !== f2 && Ef(d, f2) && (b.flags |= 32), gj(a, b), Xi(a, b, g, c), b.child;
    case 6:
      return null === a && Eg(b), null;
    case 13:
      return oj(a, b, c);
    case 4:
      return yh(b, b.stateNode.containerInfo), d = b.pendingProps, null === a ? b.child = Ug(b, null, d, c) : Xi(a, b, d, c), b.child;
    case 11:
      return d = b.type, e = b.pendingProps, e = b.elementType === d ? e : Ci(d, e), Yi(a, b, d, e, c);
    case 7:
      return Xi(a, b, b.pendingProps, c), b.child;
    case 8:
      return Xi(a, b, b.pendingProps.children, c), b.child;
    case 12:
      return Xi(a, b, b.pendingProps.children, c), b.child;
    case 10:
      a: {
        d = b.type._context;
        e = b.pendingProps;
        f2 = b.memoizedProps;
        g = e.value;
        G(Wg, d._currentValue);
        d._currentValue = g;
        if (null !== f2)
          if (He(f2.value, g)) {
            if (f2.children === e.children && !Wf.current) {
              b = Zi(a, b, c);
              break a;
            }
          } else
            for (f2 = b.child, null !== f2 && (f2.return = b); null !== f2; ) {
              var h = f2.dependencies;
              if (null !== h) {
                g = f2.child;
                for (var k2 = h.firstContext; null !== k2; ) {
                  if (k2.context === d) {
                    if (1 === f2.tag) {
                      k2 = mh(-1, c & -c);
                      k2.tag = 2;
                      var l2 = f2.updateQueue;
                      if (null !== l2) {
                        l2 = l2.shared;
                        var m2 = l2.pending;
                        null === m2 ? k2.next = k2 : (k2.next = m2.next, m2.next = k2);
                        l2.pending = k2;
                      }
                    }
                    f2.lanes |= c;
                    k2 = f2.alternate;
                    null !== k2 && (k2.lanes |= c);
                    bh(
                      f2.return,
                      c,
                      b
                    );
                    h.lanes |= c;
                    break;
                  }
                  k2 = k2.next;
                }
              } else if (10 === f2.tag)
                g = f2.type === b.type ? null : f2.child;
              else if (18 === f2.tag) {
                g = f2.return;
                if (null === g)
                  throw Error(p(341));
                g.lanes |= c;
                h = g.alternate;
                null !== h && (h.lanes |= c);
                bh(g, c, b);
                g = f2.sibling;
              } else
                g = f2.child;
              if (null !== g)
                g.return = f2;
              else
                for (g = f2; null !== g; ) {
                  if (g === b) {
                    g = null;
                    break;
                  }
                  f2 = g.sibling;
                  if (null !== f2) {
                    f2.return = g.return;
                    g = f2;
                    break;
                  }
                  g = g.return;
                }
              f2 = g;
            }
        Xi(a, b, e.children, c);
        b = b.child;
      }
      return b;
    case 9:
      return e = b.type, d = b.pendingProps.children, ch(b, c), e = eh(e), d = d(e), b.flags |= 1, Xi(a, b, d, c), b.child;
    case 14:
      return d = b.type, e = Ci(d, b.pendingProps), e = Ci(d.type, e), $i(a, b, d, e, c);
    case 15:
      return bj(a, b, b.type, b.pendingProps, c);
    case 17:
      return d = b.type, e = b.pendingProps, e = b.elementType === d ? e : Ci(d, e), ij(a, b), b.tag = 1, Zf(d) ? (a = true, cg(b)) : a = false, ch(b, c), Gi(b, d, e), Ii(b, d, e, c), jj(null, b, d, true, a, c);
    case 19:
      return xj(a, b, c);
    case 22:
      return dj(a, b, c);
  }
  throw Error(p(156, b.tag));
};
function Fk(a, b) {
  return ac(a, b);
}
function $k(a, b, c, d) {
  this.tag = a;
  this.key = c;
  this.sibling = this.child = this.return = this.stateNode = this.type = this.elementType = null;
  this.index = 0;
  this.ref = null;
  this.pendingProps = b;
  this.dependencies = this.memoizedState = this.updateQueue = this.memoizedProps = null;
  this.mode = d;
  this.subtreeFlags = this.flags = 0;
  this.deletions = null;
  this.childLanes = this.lanes = 0;
  this.alternate = null;
}
function Bg(a, b, c, d) {
  return new $k(a, b, c, d);
}
function aj(a) {
  a = a.prototype;
  return !(!a || !a.isReactComponent);
}
function Zk(a) {
  if ("function" === typeof a)
    return aj(a) ? 1 : 0;
  if (void 0 !== a && null !== a) {
    a = a.$$typeof;
    if (a === Da)
      return 11;
    if (a === Ga)
      return 14;
  }
  return 2;
}
function Pg(a, b) {
  var c = a.alternate;
  null === c ? (c = Bg(a.tag, b, a.key, a.mode), c.elementType = a.elementType, c.type = a.type, c.stateNode = a.stateNode, c.alternate = a, a.alternate = c) : (c.pendingProps = b, c.type = a.type, c.flags = 0, c.subtreeFlags = 0, c.deletions = null);
  c.flags = a.flags & 14680064;
  c.childLanes = a.childLanes;
  c.lanes = a.lanes;
  c.child = a.child;
  c.memoizedProps = a.memoizedProps;
  c.memoizedState = a.memoizedState;
  c.updateQueue = a.updateQueue;
  b = a.dependencies;
  c.dependencies = null === b ? null : { lanes: b.lanes, firstContext: b.firstContext };
  c.sibling = a.sibling;
  c.index = a.index;
  c.ref = a.ref;
  return c;
}
function Rg(a, b, c, d, e, f2) {
  var g = 2;
  d = a;
  if ("function" === typeof a)
    aj(a) && (g = 1);
  else if ("string" === typeof a)
    g = 5;
  else
    a:
      switch (a) {
        case ya:
          return Tg(c.children, e, f2, b);
        case za:
          g = 8;
          e |= 8;
          break;
        case Aa:
          return a = Bg(12, c, b, e | 2), a.elementType = Aa, a.lanes = f2, a;
        case Ea:
          return a = Bg(13, c, b, e), a.elementType = Ea, a.lanes = f2, a;
        case Fa:
          return a = Bg(19, c, b, e), a.elementType = Fa, a.lanes = f2, a;
        case Ia:
          return pj(c, e, f2, b);
        default:
          if ("object" === typeof a && null !== a)
            switch (a.$$typeof) {
              case Ba:
                g = 10;
                break a;
              case Ca:
                g = 9;
                break a;
              case Da:
                g = 11;
                break a;
              case Ga:
                g = 14;
                break a;
              case Ha:
                g = 16;
                d = null;
                break a;
            }
          throw Error(p(130, null == a ? a : typeof a, ""));
      }
  b = Bg(g, c, b, e);
  b.elementType = a;
  b.type = d;
  b.lanes = f2;
  return b;
}
function Tg(a, b, c, d) {
  a = Bg(7, a, d, b);
  a.lanes = c;
  return a;
}
function pj(a, b, c, d) {
  a = Bg(22, a, d, b);
  a.elementType = Ia;
  a.lanes = c;
  a.stateNode = { isHidden: false };
  return a;
}
function Qg(a, b, c) {
  a = Bg(6, a, null, b);
  a.lanes = c;
  return a;
}
function Sg(a, b, c) {
  b = Bg(4, null !== a.children ? a.children : [], a.key, b);
  b.lanes = c;
  b.stateNode = { containerInfo: a.containerInfo, pendingChildren: null, implementation: a.implementation };
  return b;
}
function al(a, b, c, d, e) {
  this.tag = b;
  this.containerInfo = a;
  this.finishedWork = this.pingCache = this.current = this.pendingChildren = null;
  this.timeoutHandle = -1;
  this.callbackNode = this.pendingContext = this.context = null;
  this.callbackPriority = 0;
  this.eventTimes = zc(0);
  this.expirationTimes = zc(-1);
  this.entangledLanes = this.finishedLanes = this.mutableReadLanes = this.expiredLanes = this.pingedLanes = this.suspendedLanes = this.pendingLanes = 0;
  this.entanglements = zc(0);
  this.identifierPrefix = d;
  this.onRecoverableError = e;
  this.mutableSourceEagerHydrationData = null;
}
function bl(a, b, c, d, e, f2, g, h, k2) {
  a = new al(a, b, c, h, k2);
  1 === b ? (b = 1, true === f2 && (b |= 8)) : b = 0;
  f2 = Bg(3, null, null, b);
  a.current = f2;
  f2.stateNode = a;
  f2.memoizedState = { element: d, isDehydrated: c, cache: null, transitions: null, pendingSuspenseBoundaries: null };
  kh(f2);
  return a;
}
function cl(a, b, c) {
  var d = 3 < arguments.length && void 0 !== arguments[3] ? arguments[3] : null;
  return { $$typeof: wa, key: null == d ? null : "" + d, children: a, containerInfo: b, implementation: c };
}
function dl(a) {
  if (!a)
    return Vf;
  a = a._reactInternals;
  a: {
    if (Vb(a) !== a || 1 !== a.tag)
      throw Error(p(170));
    var b = a;
    do {
      switch (b.tag) {
        case 3:
          b = b.stateNode.context;
          break a;
        case 1:
          if (Zf(b.type)) {
            b = b.stateNode.__reactInternalMemoizedMergedChildContext;
            break a;
          }
      }
      b = b.return;
    } while (null !== b);
    throw Error(p(171));
  }
  if (1 === a.tag) {
    var c = a.type;
    if (Zf(c))
      return bg(a, c, b);
  }
  return b;
}
function el(a, b, c, d, e, f2, g, h, k2) {
  a = bl(c, d, true, a, e, f2, g, h, k2);
  a.context = dl(null);
  c = a.current;
  d = R();
  e = yi(c);
  f2 = mh(d, e);
  f2.callback = void 0 !== b && null !== b ? b : null;
  nh(c, f2, e);
  a.current.lanes = e;
  Ac(a, e, d);
  Dk(a, d);
  return a;
}
function fl(a, b, c, d) {
  var e = b.current, f2 = R(), g = yi(e);
  c = dl(c);
  null === b.context ? b.context = c : b.pendingContext = c;
  b = mh(f2, g);
  b.payload = { element: a };
  d = void 0 === d ? null : d;
  null !== d && (b.callback = d);
  a = nh(e, b, g);
  null !== a && (gi(a, e, g, f2), oh(a, e, g));
  return g;
}
function gl(a) {
  a = a.current;
  if (!a.child)
    return null;
  switch (a.child.tag) {
    case 5:
      return a.child.stateNode;
    default:
      return a.child.stateNode;
  }
}
function hl(a, b) {
  a = a.memoizedState;
  if (null !== a && null !== a.dehydrated) {
    var c = a.retryLane;
    a.retryLane = 0 !== c && c < b ? c : b;
  }
}
function il(a, b) {
  hl(a, b);
  (a = a.alternate) && hl(a, b);
}
function jl() {
  return null;
}
var kl = "function" === typeof reportError ? reportError : function(a) {
  console.error(a);
};
function ll(a) {
  this._internalRoot = a;
}
ml.prototype.render = ll.prototype.render = function(a) {
  var b = this._internalRoot;
  if (null === b)
    throw Error(p(409));
  fl(a, b, null, null);
};
ml.prototype.unmount = ll.prototype.unmount = function() {
  var a = this._internalRoot;
  if (null !== a) {
    this._internalRoot = null;
    var b = a.containerInfo;
    Rk(function() {
      fl(null, a, null, null);
    });
    b[uf] = null;
  }
};
function ml(a) {
  this._internalRoot = a;
}
ml.prototype.unstable_scheduleHydration = function(a) {
  if (a) {
    var b = Hc();
    a = { blockedOn: null, target: a, priority: b };
    for (var c = 0; c < Qc.length && 0 !== b && b < Qc[c].priority; c++)
      ;
    Qc.splice(c, 0, a);
    0 === c && Vc(a);
  }
};
function nl(a) {
  return !(!a || 1 !== a.nodeType && 9 !== a.nodeType && 11 !== a.nodeType);
}
function ol(a) {
  return !(!a || 1 !== a.nodeType && 9 !== a.nodeType && 11 !== a.nodeType && (8 !== a.nodeType || " react-mount-point-unstable " !== a.nodeValue));
}
function pl() {
}
function ql(a, b, c, d, e) {
  if (e) {
    if ("function" === typeof d) {
      var f2 = d;
      d = function() {
        var a2 = gl(g);
        f2.call(a2);
      };
    }
    var g = el(b, d, a, 0, null, false, false, "", pl);
    a._reactRootContainer = g;
    a[uf] = g.current;
    sf(8 === a.nodeType ? a.parentNode : a);
    Rk();
    return g;
  }
  for (; e = a.lastChild; )
    a.removeChild(e);
  if ("function" === typeof d) {
    var h = d;
    d = function() {
      var a2 = gl(k2);
      h.call(a2);
    };
  }
  var k2 = bl(a, 0, false, null, null, false, false, "", pl);
  a._reactRootContainer = k2;
  a[uf] = k2.current;
  sf(8 === a.nodeType ? a.parentNode : a);
  Rk(function() {
    fl(b, k2, c, d);
  });
  return k2;
}
function rl(a, b, c, d, e) {
  var f2 = c._reactRootContainer;
  if (f2) {
    var g = f2;
    if ("function" === typeof e) {
      var h = e;
      e = function() {
        var a2 = gl(g);
        h.call(a2);
      };
    }
    fl(b, g, a, e);
  } else
    g = ql(c, b, a, e, d);
  return gl(g);
}
Ec = function(a) {
  switch (a.tag) {
    case 3:
      var b = a.stateNode;
      if (b.current.memoizedState.isDehydrated) {
        var c = tc(b.pendingLanes);
        0 !== c && (Cc(b, c | 1), Dk(b, B()), 0 === (K & 6) && (Gj = B() + 500, jg()));
      }
      break;
    case 13:
      Rk(function() {
        var b2 = ih(a, 1);
        if (null !== b2) {
          var c2 = R();
          gi(b2, a, 1, c2);
        }
      }), il(a, 1);
  }
};
Fc = function(a) {
  if (13 === a.tag) {
    var b = ih(a, 134217728);
    if (null !== b) {
      var c = R();
      gi(b, a, 134217728, c);
    }
    il(a, 134217728);
  }
};
Gc = function(a) {
  if (13 === a.tag) {
    var b = yi(a), c = ih(a, b);
    if (null !== c) {
      var d = R();
      gi(c, a, b, d);
    }
    il(a, b);
  }
};
Hc = function() {
  return C;
};
Ic = function(a, b) {
  var c = C;
  try {
    return C = a, b();
  } finally {
    C = c;
  }
};
yb = function(a, b, c) {
  switch (b) {
    case "input":
      bb(a, c);
      b = c.name;
      if ("radio" === c.type && null != b) {
        for (c = a; c.parentNode; )
          c = c.parentNode;
        c = c.querySelectorAll("input[name=" + JSON.stringify("" + b) + '][type="radio"]');
        for (b = 0; b < c.length; b++) {
          var d = c[b];
          if (d !== a && d.form === a.form) {
            var e = Db(d);
            if (!e)
              throw Error(p(90));
            Wa(d);
            bb(d, e);
          }
        }
      }
      break;
    case "textarea":
      ib(a, c);
      break;
    case "select":
      b = c.value, null != b && fb(a, !!c.multiple, b, false);
  }
};
Gb = Qk;
Hb = Rk;
var sl = { usingClientEntryPoint: false, Events: [Cb, ue, Db, Eb, Fb, Qk] }, tl = { findFiberByHostInstance: Wc, bundleType: 0, version: "18.3.1", rendererPackageName: "react-dom" };
var ul = { bundleType: tl.bundleType, version: tl.version, rendererPackageName: tl.rendererPackageName, rendererConfig: tl.rendererConfig, overrideHookState: null, overrideHookStateDeletePath: null, overrideHookStateRenamePath: null, overrideProps: null, overridePropsDeletePath: null, overridePropsRenamePath: null, setErrorHandler: null, setSuspenseHandler: null, scheduleUpdate: null, currentDispatcherRef: ua.ReactCurrentDispatcher, findHostInstanceByFiber: function(a) {
  a = Zb(a);
  return null === a ? null : a.stateNode;
}, findFiberByHostInstance: tl.findFiberByHostInstance || jl, findHostInstancesForRefresh: null, scheduleRefresh: null, scheduleRoot: null, setRefreshHandler: null, getCurrentFiber: null, reconcilerVersion: "18.3.1-next-f1338f8080-20240426" };
if ("undefined" !== typeof __REACT_DEVTOOLS_GLOBAL_HOOK__) {
  var vl = __REACT_DEVTOOLS_GLOBAL_HOOK__;
  if (!vl.isDisabled && vl.supportsFiber)
    try {
      kc = vl.inject(ul), lc = vl;
    } catch (a) {
    }
}
reactDom_production_min.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED = sl;
reactDom_production_min.createPortal = function(a, b) {
  var c = 2 < arguments.length && void 0 !== arguments[2] ? arguments[2] : null;
  if (!nl(b))
    throw Error(p(200));
  return cl(a, b, null, c);
};
reactDom_production_min.createRoot = function(a, b) {
  if (!nl(a))
    throw Error(p(299));
  var c = false, d = "", e = kl;
  null !== b && void 0 !== b && (true === b.unstable_strictMode && (c = true), void 0 !== b.identifierPrefix && (d = b.identifierPrefix), void 0 !== b.onRecoverableError && (e = b.onRecoverableError));
  b = bl(a, 1, false, null, null, c, false, d, e);
  a[uf] = b.current;
  sf(8 === a.nodeType ? a.parentNode : a);
  return new ll(b);
};
reactDom_production_min.findDOMNode = function(a) {
  if (null == a)
    return null;
  if (1 === a.nodeType)
    return a;
  var b = a._reactInternals;
  if (void 0 === b) {
    if ("function" === typeof a.render)
      throw Error(p(188));
    a = Object.keys(a).join(",");
    throw Error(p(268, a));
  }
  a = Zb(b);
  a = null === a ? null : a.stateNode;
  return a;
};
reactDom_production_min.flushSync = function(a) {
  return Rk(a);
};
reactDom_production_min.hydrate = function(a, b, c) {
  if (!ol(b))
    throw Error(p(200));
  return rl(null, a, b, true, c);
};
reactDom_production_min.hydrateRoot = function(a, b, c) {
  if (!nl(a))
    throw Error(p(405));
  var d = null != c && c.hydratedSources || null, e = false, f2 = "", g = kl;
  null !== c && void 0 !== c && (true === c.unstable_strictMode && (e = true), void 0 !== c.identifierPrefix && (f2 = c.identifierPrefix), void 0 !== c.onRecoverableError && (g = c.onRecoverableError));
  b = el(b, null, a, 1, null != c ? c : null, e, false, f2, g);
  a[uf] = b.current;
  sf(a);
  if (d)
    for (a = 0; a < d.length; a++)
      c = d[a], e = c._getVersion, e = e(c._source), null == b.mutableSourceEagerHydrationData ? b.mutableSourceEagerHydrationData = [c, e] : b.mutableSourceEagerHydrationData.push(
        c,
        e
      );
  return new ml(b);
};
reactDom_production_min.render = function(a, b, c) {
  if (!ol(b))
    throw Error(p(200));
  return rl(null, a, b, false, c);
};
reactDom_production_min.unmountComponentAtNode = function(a) {
  if (!ol(a))
    throw Error(p(40));
  return a._reactRootContainer ? (Rk(function() {
    rl(null, null, a, false, function() {
      a._reactRootContainer = null;
      a[uf] = null;
    });
  }), true) : false;
};
reactDom_production_min.unstable_batchedUpdates = Qk;
reactDom_production_min.unstable_renderSubtreeIntoContainer = function(a, b, c, d) {
  if (!ol(c))
    throw Error(p(200));
  if (null == a || void 0 === a._reactInternals)
    throw Error(p(38));
  return rl(a, b, c, false, d);
};
reactDom_production_min.version = "18.3.1-next-f1338f8080-20240426";
function checkDCE() {
  if (typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ === "undefined" || typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE !== "function") {
    return;
  }
  try {
    __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(checkDCE);
  } catch (err) {
    console.error(err);
  }
}
{
  checkDCE();
  reactDom.exports = reactDom_production_min;
}
var reactDomExports = reactDom.exports;
var m = reactDomExports;
{
  client.createRoot = m.createRoot;
  client.hydrateRoot = m.hydrateRoot;
}
var Subscribable = class {
  constructor() {
    this.listeners = /* @__PURE__ */ new Set();
    this.subscribe = this.subscribe.bind(this);
  }
  subscribe(listener) {
    this.listeners.add(listener);
    this.onSubscribe();
    return () => {
      this.listeners.delete(listener);
      this.onUnsubscribe();
    };
  }
  hasListeners() {
    return this.listeners.size > 0;
  }
  onSubscribe() {
  }
  onUnsubscribe() {
  }
};
var defaultTimeoutProvider = {
  // We need the wrapper function syntax below instead of direct references to
  // global setTimeout etc.
  //
  // BAD: `setTimeout: setTimeout`
  // GOOD: `setTimeout: (cb, delay) => setTimeout(cb, delay)`
  //
  // If we use direct references here, then anything that wants to spy on or
  // replace the global setTimeout (like tests) won't work since we'll already
  // have a hard reference to the original implementation at the time when this
  // file was imported.
  setTimeout: (callback, delay) => setTimeout(callback, delay),
  clearTimeout: (timeoutId) => clearTimeout(timeoutId),
  setInterval: (callback, delay) => setInterval(callback, delay),
  clearInterval: (intervalId) => clearInterval(intervalId)
};
var TimeoutManager = (_a = class {
  constructor() {
    // We cannot have TimeoutManager<T> as we must instantiate it with a concrete
    // type at app boot; and if we leave that type, then any new timer provider
    // would need to support ReturnType<typeof setTimeout>, which is infeasible.
    //
    // We settle for type safety for the TimeoutProvider type, and accept that
    // this class is unsafe internally to allow for extension.
    __privateAdd(this, _provider, defaultTimeoutProvider);
    __privateAdd(this, _providerCalled, false);
  }
  setTimeoutProvider(provider) {
    __privateSet(this, _provider, provider);
  }
  setTimeout(callback, delay) {
    return __privateGet(this, _provider).setTimeout(callback, delay);
  }
  clearTimeout(timeoutId) {
    __privateGet(this, _provider).clearTimeout(timeoutId);
  }
  setInterval(callback, delay) {
    return __privateGet(this, _provider).setInterval(callback, delay);
  }
  clearInterval(intervalId) {
    __privateGet(this, _provider).clearInterval(intervalId);
  }
}, _provider = new WeakMap(), _providerCalled = new WeakMap(), _a);
var timeoutManager = new TimeoutManager();
function systemSetTimeoutZero(callback) {
  setTimeout(callback, 0);
}
var isServer = typeof window === "undefined" || "Deno" in globalThis;
function noop() {
}
function functionalUpdate(updater, input) {
  return typeof updater === "function" ? updater(input) : updater;
}
function isValidTimeout(value) {
  return typeof value === "number" && value >= 0 && value !== Infinity;
}
function timeUntilStale(updatedAt, staleTime) {
  return Math.max(updatedAt + (staleTime || 0) - Date.now(), 0);
}
function resolveStaleTime(staleTime, query) {
  return typeof staleTime === "function" ? staleTime(query) : staleTime;
}
function resolveEnabled(enabled, query) {
  return typeof enabled === "function" ? enabled(query) : enabled;
}
function matchQuery(filters, query) {
  const {
    type = "all",
    exact,
    fetchStatus,
    predicate,
    queryKey,
    stale
  } = filters;
  if (queryKey) {
    if (exact) {
      if (query.queryHash !== hashQueryKeyByOptions(queryKey, query.options)) {
        return false;
      }
    } else if (!partialMatchKey(query.queryKey, queryKey)) {
      return false;
    }
  }
  if (type !== "all") {
    const isActive = query.isActive();
    if (type === "active" && !isActive) {
      return false;
    }
    if (type === "inactive" && isActive) {
      return false;
    }
  }
  if (typeof stale === "boolean" && query.isStale() !== stale) {
    return false;
  }
  if (fetchStatus && fetchStatus !== query.state.fetchStatus) {
    return false;
  }
  if (predicate && !predicate(query)) {
    return false;
  }
  return true;
}
function matchMutation(filters, mutation) {
  const { exact, status, predicate, mutationKey } = filters;
  if (mutationKey) {
    if (!mutation.options.mutationKey) {
      return false;
    }
    if (exact) {
      if (hashKey(mutation.options.mutationKey) !== hashKey(mutationKey)) {
        return false;
      }
    } else if (!partialMatchKey(mutation.options.mutationKey, mutationKey)) {
      return false;
    }
  }
  if (status && mutation.state.status !== status) {
    return false;
  }
  if (predicate && !predicate(mutation)) {
    return false;
  }
  return true;
}
function hashQueryKeyByOptions(queryKey, options) {
  const hashFn = (options == null ? void 0 : options.queryKeyHashFn) || hashKey;
  return hashFn(queryKey);
}
function hashKey(queryKey) {
  return JSON.stringify(
    queryKey,
    (_, val) => isPlainObject(val) ? Object.keys(val).sort().reduce((result, key) => {
      result[key] = val[key];
      return result;
    }, {}) : val
  );
}
function partialMatchKey(a, b) {
  if (a === b) {
    return true;
  }
  if (typeof a !== typeof b) {
    return false;
  }
  if (a && b && typeof a === "object" && typeof b === "object") {
    return Object.keys(b).every((key) => partialMatchKey(a[key], b[key]));
  }
  return false;
}
var hasOwn = Object.prototype.hasOwnProperty;
function replaceEqualDeep(a, b) {
  if (a === b) {
    return a;
  }
  const array = isPlainArray(a) && isPlainArray(b);
  if (!array && !(isPlainObject(a) && isPlainObject(b)))
    return b;
  const aItems = array ? a : Object.keys(a);
  const aSize = aItems.length;
  const bItems = array ? b : Object.keys(b);
  const bSize = bItems.length;
  const copy = array ? new Array(bSize) : {};
  let equalItems = 0;
  for (let i = 0; i < bSize; i++) {
    const key = array ? i : bItems[i];
    const aItem = a[key];
    const bItem = b[key];
    if (aItem === bItem) {
      copy[key] = aItem;
      if (array ? i < aSize : hasOwn.call(a, key))
        equalItems++;
      continue;
    }
    if (aItem === null || bItem === null || typeof aItem !== "object" || typeof bItem !== "object") {
      copy[key] = bItem;
      continue;
    }
    const v2 = replaceEqualDeep(aItem, bItem);
    copy[key] = v2;
    if (v2 === aItem)
      equalItems++;
  }
  return aSize === bSize && equalItems === aSize ? a : copy;
}
function shallowEqualObjects(a, b) {
  if (!b || Object.keys(a).length !== Object.keys(b).length) {
    return false;
  }
  for (const key in a) {
    if (a[key] !== b[key]) {
      return false;
    }
  }
  return true;
}
function isPlainArray(value) {
  return Array.isArray(value) && value.length === Object.keys(value).length;
}
function isPlainObject(o) {
  if (!hasObjectPrototype(o)) {
    return false;
  }
  const ctor = o.constructor;
  if (ctor === void 0) {
    return true;
  }
  const prot = ctor.prototype;
  if (!hasObjectPrototype(prot)) {
    return false;
  }
  if (!prot.hasOwnProperty("isPrototypeOf")) {
    return false;
  }
  if (Object.getPrototypeOf(o) !== Object.prototype) {
    return false;
  }
  return true;
}
function hasObjectPrototype(o) {
  return Object.prototype.toString.call(o) === "[object Object]";
}
function sleep(timeout) {
  return new Promise((resolve) => {
    timeoutManager.setTimeout(resolve, timeout);
  });
}
function replaceData(prevData, data, options) {
  if (typeof options.structuralSharing === "function") {
    return options.structuralSharing(prevData, data);
  } else if (options.structuralSharing !== false) {
    return replaceEqualDeep(prevData, data);
  }
  return data;
}
function addToEnd(items, item, max = 0) {
  const newItems = [...items, item];
  return max && newItems.length > max ? newItems.slice(1) : newItems;
}
function addToStart(items, item, max = 0) {
  const newItems = [item, ...items];
  return max && newItems.length > max ? newItems.slice(0, -1) : newItems;
}
var skipToken = Symbol();
function ensureQueryFn(options, fetchOptions) {
  if (!options.queryFn && (fetchOptions == null ? void 0 : fetchOptions.initialPromise)) {
    return () => fetchOptions.initialPromise;
  }
  if (!options.queryFn || options.queryFn === skipToken) {
    return () => Promise.reject(new Error(`Missing queryFn: '${options.queryHash}'`));
  }
  return options.queryFn;
}
function shouldThrowError(throwOnError, params) {
  if (typeof throwOnError === "function") {
    return throwOnError(...params);
  }
  return !!throwOnError;
}
var FocusManager = (_b = class extends Subscribable {
  constructor() {
    super();
    __privateAdd(this, _focused, void 0);
    __privateAdd(this, _cleanup, void 0);
    __privateAdd(this, _setup, void 0);
    __privateSet(this, _setup, (onFocus) => {
      if (!isServer && window.addEventListener) {
        const listener = () => onFocus();
        window.addEventListener("visibilitychange", listener, false);
        return () => {
          window.removeEventListener("visibilitychange", listener);
        };
      }
      return;
    });
  }
  onSubscribe() {
    if (!__privateGet(this, _cleanup)) {
      this.setEventListener(__privateGet(this, _setup));
    }
  }
  onUnsubscribe() {
    var _a2;
    if (!this.hasListeners()) {
      (_a2 = __privateGet(this, _cleanup)) == null ? void 0 : _a2.call(this);
      __privateSet(this, _cleanup, void 0);
    }
  }
  setEventListener(setup) {
    var _a2;
    __privateSet(this, _setup, setup);
    (_a2 = __privateGet(this, _cleanup)) == null ? void 0 : _a2.call(this);
    __privateSet(this, _cleanup, setup((focused) => {
      if (typeof focused === "boolean") {
        this.setFocused(focused);
      } else {
        this.onFocus();
      }
    }));
  }
  setFocused(focused) {
    const changed = __privateGet(this, _focused) !== focused;
    if (changed) {
      __privateSet(this, _focused, focused);
      this.onFocus();
    }
  }
  onFocus() {
    const isFocused = this.isFocused();
    this.listeners.forEach((listener) => {
      listener(isFocused);
    });
  }
  isFocused() {
    var _a2;
    if (typeof __privateGet(this, _focused) === "boolean") {
      return __privateGet(this, _focused);
    }
    return ((_a2 = globalThis.document) == null ? void 0 : _a2.visibilityState) !== "hidden";
  }
}, _focused = new WeakMap(), _cleanup = new WeakMap(), _setup = new WeakMap(), _b);
var focusManager = new FocusManager();
function pendingThenable() {
  let resolve;
  let reject;
  const thenable = new Promise((_resolve, _reject) => {
    resolve = _resolve;
    reject = _reject;
  });
  thenable.status = "pending";
  thenable.catch(() => {
  });
  function finalize(data) {
    Object.assign(thenable, data);
    delete thenable.resolve;
    delete thenable.reject;
  }
  thenable.resolve = (value) => {
    finalize({
      status: "fulfilled",
      value
    });
    resolve(value);
  };
  thenable.reject = (reason) => {
    finalize({
      status: "rejected",
      reason
    });
    reject(reason);
  };
  return thenable;
}
var defaultScheduler = systemSetTimeoutZero;
function createNotifyManager() {
  let queue = [];
  let transactions = 0;
  let notifyFn = (callback) => {
    callback();
  };
  let batchNotifyFn = (callback) => {
    callback();
  };
  let scheduleFn = defaultScheduler;
  const schedule = (callback) => {
    if (transactions) {
      queue.push(callback);
    } else {
      scheduleFn(() => {
        notifyFn(callback);
      });
    }
  };
  const flush = () => {
    const originalQueue = queue;
    queue = [];
    if (originalQueue.length) {
      scheduleFn(() => {
        batchNotifyFn(() => {
          originalQueue.forEach((callback) => {
            notifyFn(callback);
          });
        });
      });
    }
  };
  return {
    batch: (callback) => {
      let result;
      transactions++;
      try {
        result = callback();
      } finally {
        transactions--;
        if (!transactions) {
          flush();
        }
      }
      return result;
    },
    /**
     * All calls to the wrapped function will be batched.
     */
    batchCalls: (callback) => {
      return (...args) => {
        schedule(() => {
          callback(...args);
        });
      };
    },
    schedule,
    /**
     * Use this method to set a custom notify function.
     * This can be used to for example wrap notifications with `React.act` while running tests.
     */
    setNotifyFunction: (fn) => {
      notifyFn = fn;
    },
    /**
     * Use this method to set a custom function to batch notifications together into a single tick.
     * By default React Query will use the batch function provided by ReactDOM or React Native.
     */
    setBatchNotifyFunction: (fn) => {
      batchNotifyFn = fn;
    },
    setScheduler: (fn) => {
      scheduleFn = fn;
    }
  };
}
var notifyManager = createNotifyManager();
var OnlineManager = (_c = class extends Subscribable {
  constructor() {
    super();
    __privateAdd(this, _online, true);
    __privateAdd(this, _cleanup2, void 0);
    __privateAdd(this, _setup2, void 0);
    __privateSet(this, _setup2, (onOnline) => {
      if (!isServer && window.addEventListener) {
        const onlineListener = () => onOnline(true);
        const offlineListener = () => onOnline(false);
        window.addEventListener("online", onlineListener, false);
        window.addEventListener("offline", offlineListener, false);
        return () => {
          window.removeEventListener("online", onlineListener);
          window.removeEventListener("offline", offlineListener);
        };
      }
      return;
    });
  }
  onSubscribe() {
    if (!__privateGet(this, _cleanup2)) {
      this.setEventListener(__privateGet(this, _setup2));
    }
  }
  onUnsubscribe() {
    var _a2;
    if (!this.hasListeners()) {
      (_a2 = __privateGet(this, _cleanup2)) == null ? void 0 : _a2.call(this);
      __privateSet(this, _cleanup2, void 0);
    }
  }
  setEventListener(setup) {
    var _a2;
    __privateSet(this, _setup2, setup);
    (_a2 = __privateGet(this, _cleanup2)) == null ? void 0 : _a2.call(this);
    __privateSet(this, _cleanup2, setup(this.setOnline.bind(this)));
  }
  setOnline(online) {
    const changed = __privateGet(this, _online) !== online;
    if (changed) {
      __privateSet(this, _online, online);
      this.listeners.forEach((listener) => {
        listener(online);
      });
    }
  }
  isOnline() {
    return __privateGet(this, _online);
  }
}, _online = new WeakMap(), _cleanup2 = new WeakMap(), _setup2 = new WeakMap(), _c);
var onlineManager = new OnlineManager();
function defaultRetryDelay(failureCount) {
  return Math.min(1e3 * 2 ** failureCount, 3e4);
}
function canFetch(networkMode) {
  return (networkMode ?? "online") === "online" ? onlineManager.isOnline() : true;
}
var CancelledError = class extends Error {
  constructor(options) {
    super("CancelledError");
    this.revert = options == null ? void 0 : options.revert;
    this.silent = options == null ? void 0 : options.silent;
  }
};
function createRetryer(config) {
  let isRetryCancelled = false;
  let failureCount = 0;
  let continueFn;
  const thenable = pendingThenable();
  const isResolved = () => thenable.status !== "pending";
  const cancel = (cancelOptions) => {
    var _a2;
    if (!isResolved()) {
      const error = new CancelledError(cancelOptions);
      reject(error);
      (_a2 = config.onCancel) == null ? void 0 : _a2.call(config, error);
    }
  };
  const cancelRetry = () => {
    isRetryCancelled = true;
  };
  const continueRetry = () => {
    isRetryCancelled = false;
  };
  const canContinue = () => focusManager.isFocused() && (config.networkMode === "always" || onlineManager.isOnline()) && config.canRun();
  const canStart = () => canFetch(config.networkMode) && config.canRun();
  const resolve = (value) => {
    if (!isResolved()) {
      continueFn == null ? void 0 : continueFn();
      thenable.resolve(value);
    }
  };
  const reject = (value) => {
    if (!isResolved()) {
      continueFn == null ? void 0 : continueFn();
      thenable.reject(value);
    }
  };
  const pause = () => {
    return new Promise((continueResolve) => {
      var _a2;
      continueFn = (value) => {
        if (isResolved() || canContinue()) {
          continueResolve(value);
        }
      };
      (_a2 = config.onPause) == null ? void 0 : _a2.call(config);
    }).then(() => {
      var _a2;
      continueFn = void 0;
      if (!isResolved()) {
        (_a2 = config.onContinue) == null ? void 0 : _a2.call(config);
      }
    });
  };
  const run = () => {
    if (isResolved()) {
      return;
    }
    let promiseOrValue;
    const initialPromise = failureCount === 0 ? config.initialPromise : void 0;
    try {
      promiseOrValue = initialPromise ?? config.fn();
    } catch (error) {
      promiseOrValue = Promise.reject(error);
    }
    Promise.resolve(promiseOrValue).then(resolve).catch((error) => {
      var _a2;
      if (isResolved()) {
        return;
      }
      const retry = config.retry ?? (isServer ? 0 : 3);
      const retryDelay = config.retryDelay ?? defaultRetryDelay;
      const delay = typeof retryDelay === "function" ? retryDelay(failureCount, error) : retryDelay;
      const shouldRetry = retry === true || typeof retry === "number" && failureCount < retry || typeof retry === "function" && retry(failureCount, error);
      if (isRetryCancelled || !shouldRetry) {
        reject(error);
        return;
      }
      failureCount++;
      (_a2 = config.onFail) == null ? void 0 : _a2.call(config, failureCount, error);
      sleep(delay).then(() => {
        return canContinue() ? void 0 : pause();
      }).then(() => {
        if (isRetryCancelled) {
          reject(error);
        } else {
          run();
        }
      });
    });
  };
  return {
    promise: thenable,
    status: () => thenable.status,
    cancel,
    continue: () => {
      continueFn == null ? void 0 : continueFn();
      return thenable;
    },
    cancelRetry,
    continueRetry,
    canStart,
    start: () => {
      if (canStart()) {
        run();
      } else {
        pause().then(run);
      }
      return thenable;
    }
  };
}
var Removable = (_d = class {
  constructor() {
    __privateAdd(this, _gcTimeout, void 0);
  }
  destroy() {
    this.clearGcTimeout();
  }
  scheduleGc() {
    this.clearGcTimeout();
    if (isValidTimeout(this.gcTime)) {
      __privateSet(this, _gcTimeout, timeoutManager.setTimeout(() => {
        this.optionalRemove();
      }, this.gcTime));
    }
  }
  updateGcTime(newGcTime) {
    this.gcTime = Math.max(
      this.gcTime || 0,
      newGcTime ?? (isServer ? Infinity : 5 * 60 * 1e3)
    );
  }
  clearGcTimeout() {
    if (__privateGet(this, _gcTimeout)) {
      timeoutManager.clearTimeout(__privateGet(this, _gcTimeout));
      __privateSet(this, _gcTimeout, void 0);
    }
  }
}, _gcTimeout = new WeakMap(), _d);
var Query = (_e = class extends Removable {
  constructor(config) {
    super();
    __privateAdd(this, _dispatch);
    __privateAdd(this, _initialState, void 0);
    __privateAdd(this, _revertState, void 0);
    __privateAdd(this, _cache, void 0);
    __privateAdd(this, _client, void 0);
    __privateAdd(this, _retryer, void 0);
    __privateAdd(this, _defaultOptions, void 0);
    __privateAdd(this, _abortSignalConsumed, void 0);
    __privateSet(this, _abortSignalConsumed, false);
    __privateSet(this, _defaultOptions, config.defaultOptions);
    this.setOptions(config.options);
    this.observers = [];
    __privateSet(this, _client, config.client);
    __privateSet(this, _cache, __privateGet(this, _client).getQueryCache());
    this.queryKey = config.queryKey;
    this.queryHash = config.queryHash;
    __privateSet(this, _initialState, getDefaultState$1(this.options));
    this.state = config.state ?? __privateGet(this, _initialState);
    this.scheduleGc();
  }
  get meta() {
    return this.options.meta;
  }
  get promise() {
    var _a2;
    return (_a2 = __privateGet(this, _retryer)) == null ? void 0 : _a2.promise;
  }
  setOptions(options) {
    this.options = { ...__privateGet(this, _defaultOptions), ...options };
    this.updateGcTime(this.options.gcTime);
    if (this.state && this.state.data === void 0) {
      const defaultState = getDefaultState$1(this.options);
      if (defaultState.data !== void 0) {
        this.setState(
          successState(defaultState.data, defaultState.dataUpdatedAt)
        );
        __privateSet(this, _initialState, defaultState);
      }
    }
  }
  optionalRemove() {
    if (!this.observers.length && this.state.fetchStatus === "idle") {
      __privateGet(this, _cache).remove(this);
    }
  }
  setData(newData, options) {
    const data = replaceData(this.state.data, newData, this.options);
    __privateMethod(this, _dispatch, dispatch_fn).call(this, {
      data,
      type: "success",
      dataUpdatedAt: options == null ? void 0 : options.updatedAt,
      manual: options == null ? void 0 : options.manual
    });
    return data;
  }
  setState(state, setStateOptions) {
    __privateMethod(this, _dispatch, dispatch_fn).call(this, { type: "setState", state, setStateOptions });
  }
  cancel(options) {
    var _a2, _b2;
    const promise = (_a2 = __privateGet(this, _retryer)) == null ? void 0 : _a2.promise;
    (_b2 = __privateGet(this, _retryer)) == null ? void 0 : _b2.cancel(options);
    return promise ? promise.then(noop).catch(noop) : Promise.resolve();
  }
  destroy() {
    super.destroy();
    this.cancel({ silent: true });
  }
  reset() {
    this.destroy();
    this.setState(__privateGet(this, _initialState));
  }
  isActive() {
    return this.observers.some(
      (observer) => resolveEnabled(observer.options.enabled, this) !== false
    );
  }
  isDisabled() {
    if (this.getObserversCount() > 0) {
      return !this.isActive();
    }
    return this.options.queryFn === skipToken || this.state.dataUpdateCount + this.state.errorUpdateCount === 0;
  }
  isStatic() {
    if (this.getObserversCount() > 0) {
      return this.observers.some(
        (observer) => resolveStaleTime(observer.options.staleTime, this) === "static"
      );
    }
    return false;
  }
  isStale() {
    if (this.getObserversCount() > 0) {
      return this.observers.some(
        (observer) => observer.getCurrentResult().isStale
      );
    }
    return this.state.data === void 0 || this.state.isInvalidated;
  }
  isStaleByTime(staleTime = 0) {
    if (this.state.data === void 0) {
      return true;
    }
    if (staleTime === "static") {
      return false;
    }
    if (this.state.isInvalidated) {
      return true;
    }
    return !timeUntilStale(this.state.dataUpdatedAt, staleTime);
  }
  onFocus() {
    var _a2;
    const observer = this.observers.find((x2) => x2.shouldFetchOnWindowFocus());
    observer == null ? void 0 : observer.refetch({ cancelRefetch: false });
    (_a2 = __privateGet(this, _retryer)) == null ? void 0 : _a2.continue();
  }
  onOnline() {
    var _a2;
    const observer = this.observers.find((x2) => x2.shouldFetchOnReconnect());
    observer == null ? void 0 : observer.refetch({ cancelRefetch: false });
    (_a2 = __privateGet(this, _retryer)) == null ? void 0 : _a2.continue();
  }
  addObserver(observer) {
    if (!this.observers.includes(observer)) {
      this.observers.push(observer);
      this.clearGcTimeout();
      __privateGet(this, _cache).notify({ type: "observerAdded", query: this, observer });
    }
  }
  removeObserver(observer) {
    if (this.observers.includes(observer)) {
      this.observers = this.observers.filter((x2) => x2 !== observer);
      if (!this.observers.length) {
        if (__privateGet(this, _retryer)) {
          if (__privateGet(this, _abortSignalConsumed)) {
            __privateGet(this, _retryer).cancel({ revert: true });
          } else {
            __privateGet(this, _retryer).cancelRetry();
          }
        }
        this.scheduleGc();
      }
      __privateGet(this, _cache).notify({ type: "observerRemoved", query: this, observer });
    }
  }
  getObserversCount() {
    return this.observers.length;
  }
  invalidate() {
    if (!this.state.isInvalidated) {
      __privateMethod(this, _dispatch, dispatch_fn).call(this, { type: "invalidate" });
    }
  }
  async fetch(options, fetchOptions) {
    var _a2, _b2, _c2, _d2, _e2, _f2, _g2, _h2, _i2, _j2, _k2, _l;
    if (this.state.fetchStatus !== "idle" && // If the promise in the retyer is already rejected, we have to definitely
    // re-start the fetch; there is a chance that the query is still in a
    // pending state when that happens
    ((_a2 = __privateGet(this, _retryer)) == null ? void 0 : _a2.status()) !== "rejected") {
      if (this.state.data !== void 0 && (fetchOptions == null ? void 0 : fetchOptions.cancelRefetch)) {
        this.cancel({ silent: true });
      } else if (__privateGet(this, _retryer)) {
        __privateGet(this, _retryer).continueRetry();
        return __privateGet(this, _retryer).promise;
      }
    }
    if (options) {
      this.setOptions(options);
    }
    if (!this.options.queryFn) {
      const observer = this.observers.find((x2) => x2.options.queryFn);
      if (observer) {
        this.setOptions(observer.options);
      }
    }
    const abortController = new AbortController();
    const addSignalProperty = (object) => {
      Object.defineProperty(object, "signal", {
        enumerable: true,
        get: () => {
          __privateSet(this, _abortSignalConsumed, true);
          return abortController.signal;
        }
      });
    };
    const fetchFn = () => {
      const queryFn = ensureQueryFn(this.options, fetchOptions);
      const createQueryFnContext = () => {
        const queryFnContext2 = {
          client: __privateGet(this, _client),
          queryKey: this.queryKey,
          meta: this.meta
        };
        addSignalProperty(queryFnContext2);
        return queryFnContext2;
      };
      const queryFnContext = createQueryFnContext();
      __privateSet(this, _abortSignalConsumed, false);
      if (this.options.persister) {
        return this.options.persister(
          queryFn,
          queryFnContext,
          this
        );
      }
      return queryFn(queryFnContext);
    };
    const createFetchContext = () => {
      const context2 = {
        fetchOptions,
        options: this.options,
        queryKey: this.queryKey,
        client: __privateGet(this, _client),
        state: this.state,
        fetchFn
      };
      addSignalProperty(context2);
      return context2;
    };
    const context = createFetchContext();
    (_b2 = this.options.behavior) == null ? void 0 : _b2.onFetch(context, this);
    __privateSet(this, _revertState, this.state);
    if (this.state.fetchStatus === "idle" || this.state.fetchMeta !== ((_c2 = context.fetchOptions) == null ? void 0 : _c2.meta)) {
      __privateMethod(this, _dispatch, dispatch_fn).call(this, { type: "fetch", meta: (_d2 = context.fetchOptions) == null ? void 0 : _d2.meta });
    }
    __privateSet(this, _retryer, createRetryer({
      initialPromise: fetchOptions == null ? void 0 : fetchOptions.initialPromise,
      fn: context.fetchFn,
      onCancel: (error) => {
        if (error instanceof CancelledError && error.revert) {
          this.setState({
            ...__privateGet(this, _revertState),
            fetchStatus: "idle"
          });
        }
        abortController.abort();
      },
      onFail: (failureCount, error) => {
        __privateMethod(this, _dispatch, dispatch_fn).call(this, { type: "failed", failureCount, error });
      },
      onPause: () => {
        __privateMethod(this, _dispatch, dispatch_fn).call(this, { type: "pause" });
      },
      onContinue: () => {
        __privateMethod(this, _dispatch, dispatch_fn).call(this, { type: "continue" });
      },
      retry: context.options.retry,
      retryDelay: context.options.retryDelay,
      networkMode: context.options.networkMode,
      canRun: () => true
    }));
    try {
      const data = await __privateGet(this, _retryer).start();
      if (data === void 0) {
        if (false)
          ;
        throw new Error(`${this.queryHash} data is undefined`);
      }
      this.setData(data);
      (_f2 = (_e2 = __privateGet(this, _cache).config).onSuccess) == null ? void 0 : _f2.call(_e2, data, this);
      (_h2 = (_g2 = __privateGet(this, _cache).config).onSettled) == null ? void 0 : _h2.call(
        _g2,
        data,
        this.state.error,
        this
      );
      return data;
    } catch (error) {
      if (error instanceof CancelledError) {
        if (error.silent) {
          return __privateGet(this, _retryer).promise;
        } else if (error.revert) {
          if (this.state.data === void 0) {
            throw error;
          }
          return this.state.data;
        }
      }
      __privateMethod(this, _dispatch, dispatch_fn).call(this, {
        type: "error",
        error
      });
      (_j2 = (_i2 = __privateGet(this, _cache).config).onError) == null ? void 0 : _j2.call(
        _i2,
        error,
        this
      );
      (_l = (_k2 = __privateGet(this, _cache).config).onSettled) == null ? void 0 : _l.call(
        _k2,
        this.state.data,
        error,
        this
      );
      throw error;
    } finally {
      this.scheduleGc();
    }
  }
}, _initialState = new WeakMap(), _revertState = new WeakMap(), _cache = new WeakMap(), _client = new WeakMap(), _retryer = new WeakMap(), _defaultOptions = new WeakMap(), _abortSignalConsumed = new WeakMap(), _dispatch = new WeakSet(), dispatch_fn = function(action) {
  const reducer = (state) => {
    switch (action.type) {
      case "failed":
        return {
          ...state,
          fetchFailureCount: action.failureCount,
          fetchFailureReason: action.error
        };
      case "pause":
        return {
          ...state,
          fetchStatus: "paused"
        };
      case "continue":
        return {
          ...state,
          fetchStatus: "fetching"
        };
      case "fetch":
        return {
          ...state,
          ...fetchState(state.data, this.options),
          fetchMeta: action.meta ?? null
        };
      case "success":
        const newState = {
          ...state,
          ...successState(action.data, action.dataUpdatedAt),
          dataUpdateCount: state.dataUpdateCount + 1,
          ...!action.manual && {
            fetchStatus: "idle",
            fetchFailureCount: 0,
            fetchFailureReason: null
          }
        };
        __privateSet(this, _revertState, action.manual ? newState : void 0);
        return newState;
      case "error":
        const error = action.error;
        return {
          ...state,
          error,
          errorUpdateCount: state.errorUpdateCount + 1,
          errorUpdatedAt: Date.now(),
          fetchFailureCount: state.fetchFailureCount + 1,
          fetchFailureReason: error,
          fetchStatus: "idle",
          status: "error"
        };
      case "invalidate":
        return {
          ...state,
          isInvalidated: true
        };
      case "setState":
        return {
          ...state,
          ...action.state
        };
    }
  };
  this.state = reducer(this.state);
  notifyManager.batch(() => {
    this.observers.forEach((observer) => {
      observer.onQueryUpdate();
    });
    __privateGet(this, _cache).notify({ query: this, type: "updated", action });
  });
}, _e);
function fetchState(data, options) {
  return {
    fetchFailureCount: 0,
    fetchFailureReason: null,
    fetchStatus: canFetch(options.networkMode) ? "fetching" : "paused",
    ...data === void 0 && {
      error: null,
      status: "pending"
    }
  };
}
function successState(data, dataUpdatedAt) {
  return {
    data,
    dataUpdatedAt: dataUpdatedAt ?? Date.now(),
    error: null,
    isInvalidated: false,
    status: "success"
  };
}
function getDefaultState$1(options) {
  const data = typeof options.initialData === "function" ? options.initialData() : options.initialData;
  const hasData = data !== void 0;
  const initialDataUpdatedAt = hasData ? typeof options.initialDataUpdatedAt === "function" ? options.initialDataUpdatedAt() : options.initialDataUpdatedAt : 0;
  return {
    data,
    dataUpdateCount: 0,
    dataUpdatedAt: hasData ? initialDataUpdatedAt ?? Date.now() : 0,
    error: null,
    errorUpdateCount: 0,
    errorUpdatedAt: 0,
    fetchFailureCount: 0,
    fetchFailureReason: null,
    fetchMeta: null,
    isInvalidated: false,
    status: hasData ? "success" : "pending",
    fetchStatus: "idle"
  };
}
var QueryObserver = (_f = class extends Subscribable {
  constructor(client2, options) {
    super();
    __privateAdd(this, _executeFetch);
    __privateAdd(this, _updateStaleTimeout);
    __privateAdd(this, _computeRefetchInterval);
    __privateAdd(this, _updateRefetchInterval);
    __privateAdd(this, _updateTimers);
    __privateAdd(this, _clearStaleTimeout);
    __privateAdd(this, _clearRefetchInterval);
    __privateAdd(this, _updateQuery);
    __privateAdd(this, _notify);
    __privateAdd(this, _client2, void 0);
    __privateAdd(this, _currentQuery, void 0);
    __privateAdd(this, _currentQueryInitialState, void 0);
    __privateAdd(this, _currentResult, void 0);
    __privateAdd(this, _currentResultState, void 0);
    __privateAdd(this, _currentResultOptions, void 0);
    __privateAdd(this, _currentThenable, void 0);
    __privateAdd(this, _selectError, void 0);
    __privateAdd(this, _selectFn, void 0);
    __privateAdd(this, _selectResult, void 0);
    // This property keeps track of the last query with defined data.
    // It will be used to pass the previous data and query to the placeholder function between renders.
    __privateAdd(this, _lastQueryWithDefinedData, void 0);
    __privateAdd(this, _staleTimeoutId, void 0);
    __privateAdd(this, _refetchIntervalId, void 0);
    __privateAdd(this, _currentRefetchInterval, void 0);
    __privateAdd(this, _trackedProps, /* @__PURE__ */ new Set());
    this.options = options;
    __privateSet(this, _client2, client2);
    __privateSet(this, _selectError, null);
    __privateSet(this, _currentThenable, pendingThenable());
    this.bindMethods();
    this.setOptions(options);
  }
  bindMethods() {
    this.refetch = this.refetch.bind(this);
  }
  onSubscribe() {
    if (this.listeners.size === 1) {
      __privateGet(this, _currentQuery).addObserver(this);
      if (shouldFetchOnMount(__privateGet(this, _currentQuery), this.options)) {
        __privateMethod(this, _executeFetch, executeFetch_fn).call(this);
      } else {
        this.updateResult();
      }
      __privateMethod(this, _updateTimers, updateTimers_fn).call(this);
    }
  }
  onUnsubscribe() {
    if (!this.hasListeners()) {
      this.destroy();
    }
  }
  shouldFetchOnReconnect() {
    return shouldFetchOn(
      __privateGet(this, _currentQuery),
      this.options,
      this.options.refetchOnReconnect
    );
  }
  shouldFetchOnWindowFocus() {
    return shouldFetchOn(
      __privateGet(this, _currentQuery),
      this.options,
      this.options.refetchOnWindowFocus
    );
  }
  destroy() {
    this.listeners = /* @__PURE__ */ new Set();
    __privateMethod(this, _clearStaleTimeout, clearStaleTimeout_fn).call(this);
    __privateMethod(this, _clearRefetchInterval, clearRefetchInterval_fn).call(this);
    __privateGet(this, _currentQuery).removeObserver(this);
  }
  setOptions(options) {
    const prevOptions = this.options;
    const prevQuery = __privateGet(this, _currentQuery);
    this.options = __privateGet(this, _client2).defaultQueryOptions(options);
    if (this.options.enabled !== void 0 && typeof this.options.enabled !== "boolean" && typeof this.options.enabled !== "function" && typeof resolveEnabled(this.options.enabled, __privateGet(this, _currentQuery)) !== "boolean") {
      throw new Error(
        "Expected enabled to be a boolean or a callback that returns a boolean"
      );
    }
    __privateMethod(this, _updateQuery, updateQuery_fn).call(this);
    __privateGet(this, _currentQuery).setOptions(this.options);
    if (prevOptions._defaulted && !shallowEqualObjects(this.options, prevOptions)) {
      __privateGet(this, _client2).getQueryCache().notify({
        type: "observerOptionsUpdated",
        query: __privateGet(this, _currentQuery),
        observer: this
      });
    }
    const mounted = this.hasListeners();
    if (mounted && shouldFetchOptionally(
      __privateGet(this, _currentQuery),
      prevQuery,
      this.options,
      prevOptions
    )) {
      __privateMethod(this, _executeFetch, executeFetch_fn).call(this);
    }
    this.updateResult();
    if (mounted && (__privateGet(this, _currentQuery) !== prevQuery || resolveEnabled(this.options.enabled, __privateGet(this, _currentQuery)) !== resolveEnabled(prevOptions.enabled, __privateGet(this, _currentQuery)) || resolveStaleTime(this.options.staleTime, __privateGet(this, _currentQuery)) !== resolveStaleTime(prevOptions.staleTime, __privateGet(this, _currentQuery)))) {
      __privateMethod(this, _updateStaleTimeout, updateStaleTimeout_fn).call(this);
    }
    const nextRefetchInterval = __privateMethod(this, _computeRefetchInterval, computeRefetchInterval_fn).call(this);
    if (mounted && (__privateGet(this, _currentQuery) !== prevQuery || resolveEnabled(this.options.enabled, __privateGet(this, _currentQuery)) !== resolveEnabled(prevOptions.enabled, __privateGet(this, _currentQuery)) || nextRefetchInterval !== __privateGet(this, _currentRefetchInterval))) {
      __privateMethod(this, _updateRefetchInterval, updateRefetchInterval_fn).call(this, nextRefetchInterval);
    }
  }
  getOptimisticResult(options) {
    const query = __privateGet(this, _client2).getQueryCache().build(__privateGet(this, _client2), options);
    const result = this.createResult(query, options);
    if (shouldAssignObserverCurrentProperties(this, result)) {
      __privateSet(this, _currentResult, result);
      __privateSet(this, _currentResultOptions, this.options);
      __privateSet(this, _currentResultState, __privateGet(this, _currentQuery).state);
    }
    return result;
  }
  getCurrentResult() {
    return __privateGet(this, _currentResult);
  }
  trackResult(result, onPropTracked) {
    return new Proxy(result, {
      get: (target, key) => {
        this.trackProp(key);
        onPropTracked == null ? void 0 : onPropTracked(key);
        if (key === "promise") {
          this.trackProp("data");
          if (!this.options.experimental_prefetchInRender && __privateGet(this, _currentThenable).status === "pending") {
            __privateGet(this, _currentThenable).reject(
              new Error(
                "experimental_prefetchInRender feature flag is not enabled"
              )
            );
          }
        }
        return Reflect.get(target, key);
      }
    });
  }
  trackProp(key) {
    __privateGet(this, _trackedProps).add(key);
  }
  getCurrentQuery() {
    return __privateGet(this, _currentQuery);
  }
  refetch({ ...options } = {}) {
    return this.fetch({
      ...options
    });
  }
  fetchOptimistic(options) {
    const defaultedOptions = __privateGet(this, _client2).defaultQueryOptions(options);
    const query = __privateGet(this, _client2).getQueryCache().build(__privateGet(this, _client2), defaultedOptions);
    return query.fetch().then(() => this.createResult(query, defaultedOptions));
  }
  fetch(fetchOptions) {
    return __privateMethod(this, _executeFetch, executeFetch_fn).call(this, {
      ...fetchOptions,
      cancelRefetch: fetchOptions.cancelRefetch ?? true
    }).then(() => {
      this.updateResult();
      return __privateGet(this, _currentResult);
    });
  }
  createResult(query, options) {
    var _a2;
    const prevQuery = __privateGet(this, _currentQuery);
    const prevOptions = this.options;
    const prevResult = __privateGet(this, _currentResult);
    const prevResultState = __privateGet(this, _currentResultState);
    const prevResultOptions = __privateGet(this, _currentResultOptions);
    const queryChange = query !== prevQuery;
    const queryInitialState = queryChange ? query.state : __privateGet(this, _currentQueryInitialState);
    const { state } = query;
    let newState = { ...state };
    let isPlaceholderData = false;
    let data;
    if (options._optimisticResults) {
      const mounted = this.hasListeners();
      const fetchOnMount = !mounted && shouldFetchOnMount(query, options);
      const fetchOptionally = mounted && shouldFetchOptionally(query, prevQuery, options, prevOptions);
      if (fetchOnMount || fetchOptionally) {
        newState = {
          ...newState,
          ...fetchState(state.data, query.options)
        };
      }
      if (options._optimisticResults === "isRestoring") {
        newState.fetchStatus = "idle";
      }
    }
    let { error, errorUpdatedAt, status } = newState;
    data = newState.data;
    let skipSelect = false;
    if (options.placeholderData !== void 0 && data === void 0 && status === "pending") {
      let placeholderData;
      if ((prevResult == null ? void 0 : prevResult.isPlaceholderData) && options.placeholderData === (prevResultOptions == null ? void 0 : prevResultOptions.placeholderData)) {
        placeholderData = prevResult.data;
        skipSelect = true;
      } else {
        placeholderData = typeof options.placeholderData === "function" ? options.placeholderData(
          (_a2 = __privateGet(this, _lastQueryWithDefinedData)) == null ? void 0 : _a2.state.data,
          __privateGet(this, _lastQueryWithDefinedData)
        ) : options.placeholderData;
      }
      if (placeholderData !== void 0) {
        status = "success";
        data = replaceData(
          prevResult == null ? void 0 : prevResult.data,
          placeholderData,
          options
        );
        isPlaceholderData = true;
      }
    }
    if (options.select && data !== void 0 && !skipSelect) {
      if (prevResult && data === (prevResultState == null ? void 0 : prevResultState.data) && options.select === __privateGet(this, _selectFn)) {
        data = __privateGet(this, _selectResult);
      } else {
        try {
          __privateSet(this, _selectFn, options.select);
          data = options.select(data);
          data = replaceData(prevResult == null ? void 0 : prevResult.data, data, options);
          __privateSet(this, _selectResult, data);
          __privateSet(this, _selectError, null);
        } catch (selectError) {
          __privateSet(this, _selectError, selectError);
        }
      }
    }
    if (__privateGet(this, _selectError)) {
      error = __privateGet(this, _selectError);
      data = __privateGet(this, _selectResult);
      errorUpdatedAt = Date.now();
      status = "error";
    }
    const isFetching = newState.fetchStatus === "fetching";
    const isPending = status === "pending";
    const isError = status === "error";
    const isLoading = isPending && isFetching;
    const hasData = data !== void 0;
    const result = {
      status,
      fetchStatus: newState.fetchStatus,
      isPending,
      isSuccess: status === "success",
      isError,
      isInitialLoading: isLoading,
      isLoading,
      data,
      dataUpdatedAt: newState.dataUpdatedAt,
      error,
      errorUpdatedAt,
      failureCount: newState.fetchFailureCount,
      failureReason: newState.fetchFailureReason,
      errorUpdateCount: newState.errorUpdateCount,
      isFetched: newState.dataUpdateCount > 0 || newState.errorUpdateCount > 0,
      isFetchedAfterMount: newState.dataUpdateCount > queryInitialState.dataUpdateCount || newState.errorUpdateCount > queryInitialState.errorUpdateCount,
      isFetching,
      isRefetching: isFetching && !isPending,
      isLoadingError: isError && !hasData,
      isPaused: newState.fetchStatus === "paused",
      isPlaceholderData,
      isRefetchError: isError && hasData,
      isStale: isStale(query, options),
      refetch: this.refetch,
      promise: __privateGet(this, _currentThenable),
      isEnabled: resolveEnabled(options.enabled, query) !== false
    };
    const nextResult = result;
    if (this.options.experimental_prefetchInRender) {
      const finalizeThenableIfPossible = (thenable) => {
        if (nextResult.status === "error") {
          thenable.reject(nextResult.error);
        } else if (nextResult.data !== void 0) {
          thenable.resolve(nextResult.data);
        }
      };
      const recreateThenable = () => {
        const pending = __privateSet(this, _currentThenable, nextResult.promise = pendingThenable());
        finalizeThenableIfPossible(pending);
      };
      const prevThenable = __privateGet(this, _currentThenable);
      switch (prevThenable.status) {
        case "pending":
          if (query.queryHash === prevQuery.queryHash) {
            finalizeThenableIfPossible(prevThenable);
          }
          break;
        case "fulfilled":
          if (nextResult.status === "error" || nextResult.data !== prevThenable.value) {
            recreateThenable();
          }
          break;
        case "rejected":
          if (nextResult.status !== "error" || nextResult.error !== prevThenable.reason) {
            recreateThenable();
          }
          break;
      }
    }
    return nextResult;
  }
  updateResult() {
    const prevResult = __privateGet(this, _currentResult);
    const nextResult = this.createResult(__privateGet(this, _currentQuery), this.options);
    __privateSet(this, _currentResultState, __privateGet(this, _currentQuery).state);
    __privateSet(this, _currentResultOptions, this.options);
    if (__privateGet(this, _currentResultState).data !== void 0) {
      __privateSet(this, _lastQueryWithDefinedData, __privateGet(this, _currentQuery));
    }
    if (shallowEqualObjects(nextResult, prevResult)) {
      return;
    }
    __privateSet(this, _currentResult, nextResult);
    const shouldNotifyListeners = () => {
      if (!prevResult) {
        return true;
      }
      const { notifyOnChangeProps } = this.options;
      const notifyOnChangePropsValue = typeof notifyOnChangeProps === "function" ? notifyOnChangeProps() : notifyOnChangeProps;
      if (notifyOnChangePropsValue === "all" || !notifyOnChangePropsValue && !__privateGet(this, _trackedProps).size) {
        return true;
      }
      const includedProps = new Set(
        notifyOnChangePropsValue ?? __privateGet(this, _trackedProps)
      );
      if (this.options.throwOnError) {
        includedProps.add("error");
      }
      return Object.keys(__privateGet(this, _currentResult)).some((key) => {
        const typedKey = key;
        const changed = __privateGet(this, _currentResult)[typedKey] !== prevResult[typedKey];
        return changed && includedProps.has(typedKey);
      });
    };
    __privateMethod(this, _notify, notify_fn).call(this, { listeners: shouldNotifyListeners() });
  }
  onQueryUpdate() {
    this.updateResult();
    if (this.hasListeners()) {
      __privateMethod(this, _updateTimers, updateTimers_fn).call(this);
    }
  }
}, _client2 = new WeakMap(), _currentQuery = new WeakMap(), _currentQueryInitialState = new WeakMap(), _currentResult = new WeakMap(), _currentResultState = new WeakMap(), _currentResultOptions = new WeakMap(), _currentThenable = new WeakMap(), _selectError = new WeakMap(), _selectFn = new WeakMap(), _selectResult = new WeakMap(), _lastQueryWithDefinedData = new WeakMap(), _staleTimeoutId = new WeakMap(), _refetchIntervalId = new WeakMap(), _currentRefetchInterval = new WeakMap(), _trackedProps = new WeakMap(), _executeFetch = new WeakSet(), executeFetch_fn = function(fetchOptions) {
  __privateMethod(this, _updateQuery, updateQuery_fn).call(this);
  let promise = __privateGet(this, _currentQuery).fetch(
    this.options,
    fetchOptions
  );
  if (!(fetchOptions == null ? void 0 : fetchOptions.throwOnError)) {
    promise = promise.catch(noop);
  }
  return promise;
}, _updateStaleTimeout = new WeakSet(), updateStaleTimeout_fn = function() {
  __privateMethod(this, _clearStaleTimeout, clearStaleTimeout_fn).call(this);
  const staleTime = resolveStaleTime(
    this.options.staleTime,
    __privateGet(this, _currentQuery)
  );
  if (isServer || __privateGet(this, _currentResult).isStale || !isValidTimeout(staleTime)) {
    return;
  }
  const time = timeUntilStale(__privateGet(this, _currentResult).dataUpdatedAt, staleTime);
  const timeout = time + 1;
  __privateSet(this, _staleTimeoutId, timeoutManager.setTimeout(() => {
    if (!__privateGet(this, _currentResult).isStale) {
      this.updateResult();
    }
  }, timeout));
}, _computeRefetchInterval = new WeakSet(), computeRefetchInterval_fn = function() {
  return (typeof this.options.refetchInterval === "function" ? this.options.refetchInterval(__privateGet(this, _currentQuery)) : this.options.refetchInterval) ?? false;
}, _updateRefetchInterval = new WeakSet(), updateRefetchInterval_fn = function(nextInterval) {
  __privateMethod(this, _clearRefetchInterval, clearRefetchInterval_fn).call(this);
  __privateSet(this, _currentRefetchInterval, nextInterval);
  if (isServer || resolveEnabled(this.options.enabled, __privateGet(this, _currentQuery)) === false || !isValidTimeout(__privateGet(this, _currentRefetchInterval)) || __privateGet(this, _currentRefetchInterval) === 0) {
    return;
  }
  __privateSet(this, _refetchIntervalId, timeoutManager.setInterval(() => {
    if (this.options.refetchIntervalInBackground || focusManager.isFocused()) {
      __privateMethod(this, _executeFetch, executeFetch_fn).call(this);
    }
  }, __privateGet(this, _currentRefetchInterval)));
}, _updateTimers = new WeakSet(), updateTimers_fn = function() {
  __privateMethod(this, _updateStaleTimeout, updateStaleTimeout_fn).call(this);
  __privateMethod(this, _updateRefetchInterval, updateRefetchInterval_fn).call(this, __privateMethod(this, _computeRefetchInterval, computeRefetchInterval_fn).call(this));
}, _clearStaleTimeout = new WeakSet(), clearStaleTimeout_fn = function() {
  if (__privateGet(this, _staleTimeoutId)) {
    timeoutManager.clearTimeout(__privateGet(this, _staleTimeoutId));
    __privateSet(this, _staleTimeoutId, void 0);
  }
}, _clearRefetchInterval = new WeakSet(), clearRefetchInterval_fn = function() {
  if (__privateGet(this, _refetchIntervalId)) {
    timeoutManager.clearInterval(__privateGet(this, _refetchIntervalId));
    __privateSet(this, _refetchIntervalId, void 0);
  }
}, _updateQuery = new WeakSet(), updateQuery_fn = function() {
  const query = __privateGet(this, _client2).getQueryCache().build(__privateGet(this, _client2), this.options);
  if (query === __privateGet(this, _currentQuery)) {
    return;
  }
  const prevQuery = __privateGet(this, _currentQuery);
  __privateSet(this, _currentQuery, query);
  __privateSet(this, _currentQueryInitialState, query.state);
  if (this.hasListeners()) {
    prevQuery == null ? void 0 : prevQuery.removeObserver(this);
    query.addObserver(this);
  }
}, _notify = new WeakSet(), notify_fn = function(notifyOptions) {
  notifyManager.batch(() => {
    if (notifyOptions.listeners) {
      this.listeners.forEach((listener) => {
        listener(__privateGet(this, _currentResult));
      });
    }
    __privateGet(this, _client2).getQueryCache().notify({
      query: __privateGet(this, _currentQuery),
      type: "observerResultsUpdated"
    });
  });
}, _f);
function shouldLoadOnMount(query, options) {
  return resolveEnabled(options.enabled, query) !== false && query.state.data === void 0 && !(query.state.status === "error" && options.retryOnMount === false);
}
function shouldFetchOnMount(query, options) {
  return shouldLoadOnMount(query, options) || query.state.data !== void 0 && shouldFetchOn(query, options, options.refetchOnMount);
}
function shouldFetchOn(query, options, field) {
  if (resolveEnabled(options.enabled, query) !== false && resolveStaleTime(options.staleTime, query) !== "static") {
    const value = typeof field === "function" ? field(query) : field;
    return value === "always" || value !== false && isStale(query, options);
  }
  return false;
}
function shouldFetchOptionally(query, prevQuery, options, prevOptions) {
  return (query !== prevQuery || resolveEnabled(prevOptions.enabled, query) === false) && (!options.suspense || query.state.status !== "error") && isStale(query, options);
}
function isStale(query, options) {
  return resolveEnabled(options.enabled, query) !== false && query.isStaleByTime(resolveStaleTime(options.staleTime, query));
}
function shouldAssignObserverCurrentProperties(observer, optimisticResult) {
  if (!shallowEqualObjects(observer.getCurrentResult(), optimisticResult)) {
    return true;
  }
  return false;
}
function infiniteQueryBehavior(pages) {
  return {
    onFetch: (context, query) => {
      var _a2, _b2, _c2, _d2, _e2;
      const options = context.options;
      const direction = (_c2 = (_b2 = (_a2 = context.fetchOptions) == null ? void 0 : _a2.meta) == null ? void 0 : _b2.fetchMore) == null ? void 0 : _c2.direction;
      const oldPages = ((_d2 = context.state.data) == null ? void 0 : _d2.pages) || [];
      const oldPageParams = ((_e2 = context.state.data) == null ? void 0 : _e2.pageParams) || [];
      let result = { pages: [], pageParams: [] };
      let currentPage = 0;
      const fetchFn = async () => {
        let cancelled = false;
        const addSignalProperty = (object) => {
          Object.defineProperty(object, "signal", {
            enumerable: true,
            get: () => {
              if (context.signal.aborted) {
                cancelled = true;
              } else {
                context.signal.addEventListener("abort", () => {
                  cancelled = true;
                });
              }
              return context.signal;
            }
          });
        };
        const queryFn = ensureQueryFn(context.options, context.fetchOptions);
        const fetchPage = async (data, param, previous) => {
          if (cancelled) {
            return Promise.reject();
          }
          if (param == null && data.pages.length) {
            return Promise.resolve(data);
          }
          const createQueryFnContext = () => {
            const queryFnContext2 = {
              client: context.client,
              queryKey: context.queryKey,
              pageParam: param,
              direction: previous ? "backward" : "forward",
              meta: context.options.meta
            };
            addSignalProperty(queryFnContext2);
            return queryFnContext2;
          };
          const queryFnContext = createQueryFnContext();
          const page = await queryFn(queryFnContext);
          const { maxPages } = context.options;
          const addTo = previous ? addToStart : addToEnd;
          return {
            pages: addTo(data.pages, page, maxPages),
            pageParams: addTo(data.pageParams, param, maxPages)
          };
        };
        if (direction && oldPages.length) {
          const previous = direction === "backward";
          const pageParamFn = previous ? getPreviousPageParam : getNextPageParam;
          const oldData = {
            pages: oldPages,
            pageParams: oldPageParams
          };
          const param = pageParamFn(options, oldData);
          result = await fetchPage(oldData, param, previous);
        } else {
          const remainingPages = pages ?? oldPages.length;
          do {
            const param = currentPage === 0 ? oldPageParams[0] ?? options.initialPageParam : getNextPageParam(options, result);
            if (currentPage > 0 && param == null) {
              break;
            }
            result = await fetchPage(result, param);
            currentPage++;
          } while (currentPage < remainingPages);
        }
        return result;
      };
      if (context.options.persister) {
        context.fetchFn = () => {
          var _a3, _b3;
          return (_b3 = (_a3 = context.options).persister) == null ? void 0 : _b3.call(
            _a3,
            fetchFn,
            {
              client: context.client,
              queryKey: context.queryKey,
              meta: context.options.meta,
              signal: context.signal
            },
            query
          );
        };
      } else {
        context.fetchFn = fetchFn;
      }
    }
  };
}
function getNextPageParam(options, { pages, pageParams }) {
  const lastIndex = pages.length - 1;
  return pages.length > 0 ? options.getNextPageParam(
    pages[lastIndex],
    pages,
    pageParams[lastIndex],
    pageParams
  ) : void 0;
}
function getPreviousPageParam(options, { pages, pageParams }) {
  var _a2;
  return pages.length > 0 ? (_a2 = options.getPreviousPageParam) == null ? void 0 : _a2.call(options, pages[0], pages, pageParams[0], pageParams) : void 0;
}
var Mutation = (_g = class extends Removable {
  constructor(config) {
    super();
    __privateAdd(this, _dispatch2);
    __privateAdd(this, _client3, void 0);
    __privateAdd(this, _observers, void 0);
    __privateAdd(this, _mutationCache, void 0);
    __privateAdd(this, _retryer2, void 0);
    __privateSet(this, _client3, config.client);
    this.mutationId = config.mutationId;
    __privateSet(this, _mutationCache, config.mutationCache);
    __privateSet(this, _observers, []);
    this.state = config.state || getDefaultState();
    this.setOptions(config.options);
    this.scheduleGc();
  }
  setOptions(options) {
    this.options = options;
    this.updateGcTime(this.options.gcTime);
  }
  get meta() {
    return this.options.meta;
  }
  addObserver(observer) {
    if (!__privateGet(this, _observers).includes(observer)) {
      __privateGet(this, _observers).push(observer);
      this.clearGcTimeout();
      __privateGet(this, _mutationCache).notify({
        type: "observerAdded",
        mutation: this,
        observer
      });
    }
  }
  removeObserver(observer) {
    __privateSet(this, _observers, __privateGet(this, _observers).filter((x2) => x2 !== observer));
    this.scheduleGc();
    __privateGet(this, _mutationCache).notify({
      type: "observerRemoved",
      mutation: this,
      observer
    });
  }
  optionalRemove() {
    if (!__privateGet(this, _observers).length) {
      if (this.state.status === "pending") {
        this.scheduleGc();
      } else {
        __privateGet(this, _mutationCache).remove(this);
      }
    }
  }
  continue() {
    var _a2;
    return ((_a2 = __privateGet(this, _retryer2)) == null ? void 0 : _a2.continue()) ?? // continuing a mutation assumes that variables are set, mutation must have been dehydrated before
    this.execute(this.state.variables);
  }
  async execute(variables) {
    var _a2, _b2, _c2, _d2, _e2, _f2, _g2, _h2, _i2, _j2, _k2, _l, _m, _n, _o, _p, _q, _r, _s, _t;
    const onContinue = () => {
      __privateMethod(this, _dispatch2, dispatch_fn2).call(this, { type: "continue" });
    };
    const mutationFnContext = {
      client: __privateGet(this, _client3),
      meta: this.options.meta,
      mutationKey: this.options.mutationKey
    };
    __privateSet(this, _retryer2, createRetryer({
      fn: () => {
        if (!this.options.mutationFn) {
          return Promise.reject(new Error("No mutationFn found"));
        }
        return this.options.mutationFn(variables, mutationFnContext);
      },
      onFail: (failureCount, error) => {
        __privateMethod(this, _dispatch2, dispatch_fn2).call(this, { type: "failed", failureCount, error });
      },
      onPause: () => {
        __privateMethod(this, _dispatch2, dispatch_fn2).call(this, { type: "pause" });
      },
      onContinue,
      retry: this.options.retry ?? 0,
      retryDelay: this.options.retryDelay,
      networkMode: this.options.networkMode,
      canRun: () => __privateGet(this, _mutationCache).canRun(this)
    }));
    const restored = this.state.status === "pending";
    const isPaused = !__privateGet(this, _retryer2).canStart();
    try {
      if (restored) {
        onContinue();
      } else {
        __privateMethod(this, _dispatch2, dispatch_fn2).call(this, { type: "pending", variables, isPaused });
        await ((_b2 = (_a2 = __privateGet(this, _mutationCache).config).onMutate) == null ? void 0 : _b2.call(
          _a2,
          variables,
          this,
          mutationFnContext
        ));
        const context = await ((_d2 = (_c2 = this.options).onMutate) == null ? void 0 : _d2.call(
          _c2,
          variables,
          mutationFnContext
        ));
        if (context !== this.state.context) {
          __privateMethod(this, _dispatch2, dispatch_fn2).call(this, {
            type: "pending",
            context,
            variables,
            isPaused
          });
        }
      }
      const data = await __privateGet(this, _retryer2).start();
      await ((_f2 = (_e2 = __privateGet(this, _mutationCache).config).onSuccess) == null ? void 0 : _f2.call(
        _e2,
        data,
        variables,
        this.state.context,
        this,
        mutationFnContext
      ));
      await ((_h2 = (_g2 = this.options).onSuccess) == null ? void 0 : _h2.call(
        _g2,
        data,
        variables,
        this.state.context,
        mutationFnContext
      ));
      await ((_j2 = (_i2 = __privateGet(this, _mutationCache).config).onSettled) == null ? void 0 : _j2.call(
        _i2,
        data,
        null,
        this.state.variables,
        this.state.context,
        this,
        mutationFnContext
      ));
      await ((_l = (_k2 = this.options).onSettled) == null ? void 0 : _l.call(
        _k2,
        data,
        null,
        variables,
        this.state.context,
        mutationFnContext
      ));
      __privateMethod(this, _dispatch2, dispatch_fn2).call(this, { type: "success", data });
      return data;
    } catch (error) {
      try {
        await ((_n = (_m = __privateGet(this, _mutationCache).config).onError) == null ? void 0 : _n.call(
          _m,
          error,
          variables,
          this.state.context,
          this,
          mutationFnContext
        ));
        await ((_p = (_o = this.options).onError) == null ? void 0 : _p.call(
          _o,
          error,
          variables,
          this.state.context,
          mutationFnContext
        ));
        await ((_r = (_q = __privateGet(this, _mutationCache).config).onSettled) == null ? void 0 : _r.call(
          _q,
          void 0,
          error,
          this.state.variables,
          this.state.context,
          this,
          mutationFnContext
        ));
        await ((_t = (_s = this.options).onSettled) == null ? void 0 : _t.call(
          _s,
          void 0,
          error,
          variables,
          this.state.context,
          mutationFnContext
        ));
        throw error;
      } finally {
        __privateMethod(this, _dispatch2, dispatch_fn2).call(this, { type: "error", error });
      }
    } finally {
      __privateGet(this, _mutationCache).runNext(this);
    }
  }
}, _client3 = new WeakMap(), _observers = new WeakMap(), _mutationCache = new WeakMap(), _retryer2 = new WeakMap(), _dispatch2 = new WeakSet(), dispatch_fn2 = function(action) {
  const reducer = (state) => {
    switch (action.type) {
      case "failed":
        return {
          ...state,
          failureCount: action.failureCount,
          failureReason: action.error
        };
      case "pause":
        return {
          ...state,
          isPaused: true
        };
      case "continue":
        return {
          ...state,
          isPaused: false
        };
      case "pending":
        return {
          ...state,
          context: action.context,
          data: void 0,
          failureCount: 0,
          failureReason: null,
          error: null,
          isPaused: action.isPaused,
          status: "pending",
          variables: action.variables,
          submittedAt: Date.now()
        };
      case "success":
        return {
          ...state,
          data: action.data,
          failureCount: 0,
          failureReason: null,
          error: null,
          status: "success",
          isPaused: false
        };
      case "error":
        return {
          ...state,
          data: void 0,
          error: action.error,
          failureCount: state.failureCount + 1,
          failureReason: action.error,
          isPaused: false,
          status: "error"
        };
    }
  };
  this.state = reducer(this.state);
  notifyManager.batch(() => {
    __privateGet(this, _observers).forEach((observer) => {
      observer.onMutationUpdate(action);
    });
    __privateGet(this, _mutationCache).notify({
      mutation: this,
      type: "updated",
      action
    });
  });
}, _g);
function getDefaultState() {
  return {
    context: void 0,
    data: void 0,
    error: null,
    failureCount: 0,
    failureReason: null,
    isPaused: false,
    status: "idle",
    variables: void 0,
    submittedAt: 0
  };
}
var MutationCache = (_h = class extends Subscribable {
  constructor(config = {}) {
    super();
    __privateAdd(this, _mutations, void 0);
    __privateAdd(this, _scopes, void 0);
    __privateAdd(this, _mutationId, void 0);
    this.config = config;
    __privateSet(this, _mutations, /* @__PURE__ */ new Set());
    __privateSet(this, _scopes, /* @__PURE__ */ new Map());
    __privateSet(this, _mutationId, 0);
  }
  build(client2, options, state) {
    const mutation = new Mutation({
      client: client2,
      mutationCache: this,
      mutationId: ++__privateWrapper(this, _mutationId)._,
      options: client2.defaultMutationOptions(options),
      state
    });
    this.add(mutation);
    return mutation;
  }
  add(mutation) {
    __privateGet(this, _mutations).add(mutation);
    const scope = scopeFor(mutation);
    if (typeof scope === "string") {
      const scopedMutations = __privateGet(this, _scopes).get(scope);
      if (scopedMutations) {
        scopedMutations.push(mutation);
      } else {
        __privateGet(this, _scopes).set(scope, [mutation]);
      }
    }
    this.notify({ type: "added", mutation });
  }
  remove(mutation) {
    if (__privateGet(this, _mutations).delete(mutation)) {
      const scope = scopeFor(mutation);
      if (typeof scope === "string") {
        const scopedMutations = __privateGet(this, _scopes).get(scope);
        if (scopedMutations) {
          if (scopedMutations.length > 1) {
            const index2 = scopedMutations.indexOf(mutation);
            if (index2 !== -1) {
              scopedMutations.splice(index2, 1);
            }
          } else if (scopedMutations[0] === mutation) {
            __privateGet(this, _scopes).delete(scope);
          }
        }
      }
    }
    this.notify({ type: "removed", mutation });
  }
  canRun(mutation) {
    const scope = scopeFor(mutation);
    if (typeof scope === "string") {
      const mutationsWithSameScope = __privateGet(this, _scopes).get(scope);
      const firstPendingMutation = mutationsWithSameScope == null ? void 0 : mutationsWithSameScope.find(
        (m2) => m2.state.status === "pending"
      );
      return !firstPendingMutation || firstPendingMutation === mutation;
    } else {
      return true;
    }
  }
  runNext(mutation) {
    var _a2;
    const scope = scopeFor(mutation);
    if (typeof scope === "string") {
      const foundMutation = (_a2 = __privateGet(this, _scopes).get(scope)) == null ? void 0 : _a2.find((m2) => m2 !== mutation && m2.state.isPaused);
      return (foundMutation == null ? void 0 : foundMutation.continue()) ?? Promise.resolve();
    } else {
      return Promise.resolve();
    }
  }
  clear() {
    notifyManager.batch(() => {
      __privateGet(this, _mutations).forEach((mutation) => {
        this.notify({ type: "removed", mutation });
      });
      __privateGet(this, _mutations).clear();
      __privateGet(this, _scopes).clear();
    });
  }
  getAll() {
    return Array.from(__privateGet(this, _mutations));
  }
  find(filters) {
    const defaultedFilters = { exact: true, ...filters };
    return this.getAll().find(
      (mutation) => matchMutation(defaultedFilters, mutation)
    );
  }
  findAll(filters = {}) {
    return this.getAll().filter((mutation) => matchMutation(filters, mutation));
  }
  notify(event) {
    notifyManager.batch(() => {
      this.listeners.forEach((listener) => {
        listener(event);
      });
    });
  }
  resumePausedMutations() {
    const pausedMutations = this.getAll().filter((x2) => x2.state.isPaused);
    return notifyManager.batch(
      () => Promise.all(
        pausedMutations.map((mutation) => mutation.continue().catch(noop))
      )
    );
  }
}, _mutations = new WeakMap(), _scopes = new WeakMap(), _mutationId = new WeakMap(), _h);
function scopeFor(mutation) {
  var _a2;
  return (_a2 = mutation.options.scope) == null ? void 0 : _a2.id;
}
var MutationObserver$1 = (_i = class extends Subscribable {
  constructor(client2, options) {
    super();
    __privateAdd(this, _updateResult);
    __privateAdd(this, _notify2);
    __privateAdd(this, _client4, void 0);
    __privateAdd(this, _currentResult2, void 0);
    __privateAdd(this, _currentMutation, void 0);
    __privateAdd(this, _mutateOptions, void 0);
    __privateSet(this, _client4, client2);
    this.setOptions(options);
    this.bindMethods();
    __privateMethod(this, _updateResult, updateResult_fn).call(this);
  }
  bindMethods() {
    this.mutate = this.mutate.bind(this);
    this.reset = this.reset.bind(this);
  }
  setOptions(options) {
    var _a2;
    const prevOptions = this.options;
    this.options = __privateGet(this, _client4).defaultMutationOptions(options);
    if (!shallowEqualObjects(this.options, prevOptions)) {
      __privateGet(this, _client4).getMutationCache().notify({
        type: "observerOptionsUpdated",
        mutation: __privateGet(this, _currentMutation),
        observer: this
      });
    }
    if ((prevOptions == null ? void 0 : prevOptions.mutationKey) && this.options.mutationKey && hashKey(prevOptions.mutationKey) !== hashKey(this.options.mutationKey)) {
      this.reset();
    } else if (((_a2 = __privateGet(this, _currentMutation)) == null ? void 0 : _a2.state.status) === "pending") {
      __privateGet(this, _currentMutation).setOptions(this.options);
    }
  }
  onUnsubscribe() {
    var _a2;
    if (!this.hasListeners()) {
      (_a2 = __privateGet(this, _currentMutation)) == null ? void 0 : _a2.removeObserver(this);
    }
  }
  onMutationUpdate(action) {
    __privateMethod(this, _updateResult, updateResult_fn).call(this);
    __privateMethod(this, _notify2, notify_fn2).call(this, action);
  }
  getCurrentResult() {
    return __privateGet(this, _currentResult2);
  }
  reset() {
    var _a2;
    (_a2 = __privateGet(this, _currentMutation)) == null ? void 0 : _a2.removeObserver(this);
    __privateSet(this, _currentMutation, void 0);
    __privateMethod(this, _updateResult, updateResult_fn).call(this);
    __privateMethod(this, _notify2, notify_fn2).call(this);
  }
  mutate(variables, options) {
    var _a2;
    __privateSet(this, _mutateOptions, options);
    (_a2 = __privateGet(this, _currentMutation)) == null ? void 0 : _a2.removeObserver(this);
    __privateSet(this, _currentMutation, __privateGet(this, _client4).getMutationCache().build(__privateGet(this, _client4), this.options));
    __privateGet(this, _currentMutation).addObserver(this);
    return __privateGet(this, _currentMutation).execute(variables);
  }
}, _client4 = new WeakMap(), _currentResult2 = new WeakMap(), _currentMutation = new WeakMap(), _mutateOptions = new WeakMap(), _updateResult = new WeakSet(), updateResult_fn = function() {
  var _a2;
  const state = ((_a2 = __privateGet(this, _currentMutation)) == null ? void 0 : _a2.state) ?? getDefaultState();
  __privateSet(this, _currentResult2, {
    ...state,
    isPending: state.status === "pending",
    isSuccess: state.status === "success",
    isError: state.status === "error",
    isIdle: state.status === "idle",
    mutate: this.mutate,
    reset: this.reset
  });
}, _notify2 = new WeakSet(), notify_fn2 = function(action) {
  notifyManager.batch(() => {
    var _a2, _b2, _c2, _d2, _e2, _f2, _g2, _h2;
    if (__privateGet(this, _mutateOptions) && this.hasListeners()) {
      const variables = __privateGet(this, _currentResult2).variables;
      const onMutateResult = __privateGet(this, _currentResult2).context;
      const context = {
        client: __privateGet(this, _client4),
        meta: this.options.meta,
        mutationKey: this.options.mutationKey
      };
      if ((action == null ? void 0 : action.type) === "success") {
        (_b2 = (_a2 = __privateGet(this, _mutateOptions)).onSuccess) == null ? void 0 : _b2.call(
          _a2,
          action.data,
          variables,
          onMutateResult,
          context
        );
        (_d2 = (_c2 = __privateGet(this, _mutateOptions)).onSettled) == null ? void 0 : _d2.call(
          _c2,
          action.data,
          null,
          variables,
          onMutateResult,
          context
        );
      } else if ((action == null ? void 0 : action.type) === "error") {
        (_f2 = (_e2 = __privateGet(this, _mutateOptions)).onError) == null ? void 0 : _f2.call(
          _e2,
          action.error,
          variables,
          onMutateResult,
          context
        );
        (_h2 = (_g2 = __privateGet(this, _mutateOptions)).onSettled) == null ? void 0 : _h2.call(
          _g2,
          void 0,
          action.error,
          variables,
          onMutateResult,
          context
        );
      }
    }
    this.listeners.forEach((listener) => {
      listener(__privateGet(this, _currentResult2));
    });
  });
}, _i);
var QueryCache = (_j = class extends Subscribable {
  constructor(config = {}) {
    super();
    __privateAdd(this, _queries, void 0);
    this.config = config;
    __privateSet(this, _queries, /* @__PURE__ */ new Map());
  }
  build(client2, options, state) {
    const queryKey = options.queryKey;
    const queryHash = options.queryHash ?? hashQueryKeyByOptions(queryKey, options);
    let query = this.get(queryHash);
    if (!query) {
      query = new Query({
        client: client2,
        queryKey,
        queryHash,
        options: client2.defaultQueryOptions(options),
        state,
        defaultOptions: client2.getQueryDefaults(queryKey)
      });
      this.add(query);
    }
    return query;
  }
  add(query) {
    if (!__privateGet(this, _queries).has(query.queryHash)) {
      __privateGet(this, _queries).set(query.queryHash, query);
      this.notify({
        type: "added",
        query
      });
    }
  }
  remove(query) {
    const queryInMap = __privateGet(this, _queries).get(query.queryHash);
    if (queryInMap) {
      query.destroy();
      if (queryInMap === query) {
        __privateGet(this, _queries).delete(query.queryHash);
      }
      this.notify({ type: "removed", query });
    }
  }
  clear() {
    notifyManager.batch(() => {
      this.getAll().forEach((query) => {
        this.remove(query);
      });
    });
  }
  get(queryHash) {
    return __privateGet(this, _queries).get(queryHash);
  }
  getAll() {
    return [...__privateGet(this, _queries).values()];
  }
  find(filters) {
    const defaultedFilters = { exact: true, ...filters };
    return this.getAll().find(
      (query) => matchQuery(defaultedFilters, query)
    );
  }
  findAll(filters = {}) {
    const queries = this.getAll();
    return Object.keys(filters).length > 0 ? queries.filter((query) => matchQuery(filters, query)) : queries;
  }
  notify(event) {
    notifyManager.batch(() => {
      this.listeners.forEach((listener) => {
        listener(event);
      });
    });
  }
  onFocus() {
    notifyManager.batch(() => {
      this.getAll().forEach((query) => {
        query.onFocus();
      });
    });
  }
  onOnline() {
    notifyManager.batch(() => {
      this.getAll().forEach((query) => {
        query.onOnline();
      });
    });
  }
}, _queries = new WeakMap(), _j);
var QueryClient = (_k = class {
  constructor(config = {}) {
    __privateAdd(this, _queryCache, void 0);
    __privateAdd(this, _mutationCache2, void 0);
    __privateAdd(this, _defaultOptions2, void 0);
    __privateAdd(this, _queryDefaults, void 0);
    __privateAdd(this, _mutationDefaults, void 0);
    __privateAdd(this, _mountCount, void 0);
    __privateAdd(this, _unsubscribeFocus, void 0);
    __privateAdd(this, _unsubscribeOnline, void 0);
    __privateSet(this, _queryCache, config.queryCache || new QueryCache());
    __privateSet(this, _mutationCache2, config.mutationCache || new MutationCache());
    __privateSet(this, _defaultOptions2, config.defaultOptions || {});
    __privateSet(this, _queryDefaults, /* @__PURE__ */ new Map());
    __privateSet(this, _mutationDefaults, /* @__PURE__ */ new Map());
    __privateSet(this, _mountCount, 0);
  }
  mount() {
    __privateWrapper(this, _mountCount)._++;
    if (__privateGet(this, _mountCount) !== 1)
      return;
    __privateSet(this, _unsubscribeFocus, focusManager.subscribe(async (focused) => {
      if (focused) {
        await this.resumePausedMutations();
        __privateGet(this, _queryCache).onFocus();
      }
    }));
    __privateSet(this, _unsubscribeOnline, onlineManager.subscribe(async (online) => {
      if (online) {
        await this.resumePausedMutations();
        __privateGet(this, _queryCache).onOnline();
      }
    }));
  }
  unmount() {
    var _a2, _b2;
    __privateWrapper(this, _mountCount)._--;
    if (__privateGet(this, _mountCount) !== 0)
      return;
    (_a2 = __privateGet(this, _unsubscribeFocus)) == null ? void 0 : _a2.call(this);
    __privateSet(this, _unsubscribeFocus, void 0);
    (_b2 = __privateGet(this, _unsubscribeOnline)) == null ? void 0 : _b2.call(this);
    __privateSet(this, _unsubscribeOnline, void 0);
  }
  isFetching(filters) {
    return __privateGet(this, _queryCache).findAll({ ...filters, fetchStatus: "fetching" }).length;
  }
  isMutating(filters) {
    return __privateGet(this, _mutationCache2).findAll({ ...filters, status: "pending" }).length;
  }
  /**
   * Imperative (non-reactive) way to retrieve data for a QueryKey.
   * Should only be used in callbacks or functions where reading the latest data is necessary, e.g. for optimistic updates.
   *
   * Hint: Do not use this function inside a component, because it won't receive updates.
   * Use `useQuery` to create a `QueryObserver` that subscribes to changes.
   */
  getQueryData(queryKey) {
    var _a2;
    const options = this.defaultQueryOptions({ queryKey });
    return (_a2 = __privateGet(this, _queryCache).get(options.queryHash)) == null ? void 0 : _a2.state.data;
  }
  ensureQueryData(options) {
    const defaultedOptions = this.defaultQueryOptions(options);
    const query = __privateGet(this, _queryCache).build(this, defaultedOptions);
    const cachedData = query.state.data;
    if (cachedData === void 0) {
      return this.fetchQuery(options);
    }
    if (options.revalidateIfStale && query.isStaleByTime(resolveStaleTime(defaultedOptions.staleTime, query))) {
      void this.prefetchQuery(defaultedOptions);
    }
    return Promise.resolve(cachedData);
  }
  getQueriesData(filters) {
    return __privateGet(this, _queryCache).findAll(filters).map(({ queryKey, state }) => {
      const data = state.data;
      return [queryKey, data];
    });
  }
  setQueryData(queryKey, updater, options) {
    const defaultedOptions = this.defaultQueryOptions({ queryKey });
    const query = __privateGet(this, _queryCache).get(
      defaultedOptions.queryHash
    );
    const prevData = query == null ? void 0 : query.state.data;
    const data = functionalUpdate(updater, prevData);
    if (data === void 0) {
      return void 0;
    }
    return __privateGet(this, _queryCache).build(this, defaultedOptions).setData(data, { ...options, manual: true });
  }
  setQueriesData(filters, updater, options) {
    return notifyManager.batch(
      () => __privateGet(this, _queryCache).findAll(filters).map(({ queryKey }) => [
        queryKey,
        this.setQueryData(queryKey, updater, options)
      ])
    );
  }
  getQueryState(queryKey) {
    var _a2;
    const options = this.defaultQueryOptions({ queryKey });
    return (_a2 = __privateGet(this, _queryCache).get(
      options.queryHash
    )) == null ? void 0 : _a2.state;
  }
  removeQueries(filters) {
    const queryCache = __privateGet(this, _queryCache);
    notifyManager.batch(() => {
      queryCache.findAll(filters).forEach((query) => {
        queryCache.remove(query);
      });
    });
  }
  resetQueries(filters, options) {
    const queryCache = __privateGet(this, _queryCache);
    return notifyManager.batch(() => {
      queryCache.findAll(filters).forEach((query) => {
        query.reset();
      });
      return this.refetchQueries(
        {
          type: "active",
          ...filters
        },
        options
      );
    });
  }
  cancelQueries(filters, cancelOptions = {}) {
    const defaultedCancelOptions = { revert: true, ...cancelOptions };
    const promises = notifyManager.batch(
      () => __privateGet(this, _queryCache).findAll(filters).map((query) => query.cancel(defaultedCancelOptions))
    );
    return Promise.all(promises).then(noop).catch(noop);
  }
  invalidateQueries(filters, options = {}) {
    return notifyManager.batch(() => {
      __privateGet(this, _queryCache).findAll(filters).forEach((query) => {
        query.invalidate();
      });
      if ((filters == null ? void 0 : filters.refetchType) === "none") {
        return Promise.resolve();
      }
      return this.refetchQueries(
        {
          ...filters,
          type: (filters == null ? void 0 : filters.refetchType) ?? (filters == null ? void 0 : filters.type) ?? "active"
        },
        options
      );
    });
  }
  refetchQueries(filters, options = {}) {
    const fetchOptions = {
      ...options,
      cancelRefetch: options.cancelRefetch ?? true
    };
    const promises = notifyManager.batch(
      () => __privateGet(this, _queryCache).findAll(filters).filter((query) => !query.isDisabled() && !query.isStatic()).map((query) => {
        let promise = query.fetch(void 0, fetchOptions);
        if (!fetchOptions.throwOnError) {
          promise = promise.catch(noop);
        }
        return query.state.fetchStatus === "paused" ? Promise.resolve() : promise;
      })
    );
    return Promise.all(promises).then(noop);
  }
  fetchQuery(options) {
    const defaultedOptions = this.defaultQueryOptions(options);
    if (defaultedOptions.retry === void 0) {
      defaultedOptions.retry = false;
    }
    const query = __privateGet(this, _queryCache).build(this, defaultedOptions);
    return query.isStaleByTime(
      resolveStaleTime(defaultedOptions.staleTime, query)
    ) ? query.fetch(defaultedOptions) : Promise.resolve(query.state.data);
  }
  prefetchQuery(options) {
    return this.fetchQuery(options).then(noop).catch(noop);
  }
  fetchInfiniteQuery(options) {
    options.behavior = infiniteQueryBehavior(options.pages);
    return this.fetchQuery(options);
  }
  prefetchInfiniteQuery(options) {
    return this.fetchInfiniteQuery(options).then(noop).catch(noop);
  }
  ensureInfiniteQueryData(options) {
    options.behavior = infiniteQueryBehavior(options.pages);
    return this.ensureQueryData(options);
  }
  resumePausedMutations() {
    if (onlineManager.isOnline()) {
      return __privateGet(this, _mutationCache2).resumePausedMutations();
    }
    return Promise.resolve();
  }
  getQueryCache() {
    return __privateGet(this, _queryCache);
  }
  getMutationCache() {
    return __privateGet(this, _mutationCache2);
  }
  getDefaultOptions() {
    return __privateGet(this, _defaultOptions2);
  }
  setDefaultOptions(options) {
    __privateSet(this, _defaultOptions2, options);
  }
  setQueryDefaults(queryKey, options) {
    __privateGet(this, _queryDefaults).set(hashKey(queryKey), {
      queryKey,
      defaultOptions: options
    });
  }
  getQueryDefaults(queryKey) {
    const defaults = [...__privateGet(this, _queryDefaults).values()];
    const result = {};
    defaults.forEach((queryDefault) => {
      if (partialMatchKey(queryKey, queryDefault.queryKey)) {
        Object.assign(result, queryDefault.defaultOptions);
      }
    });
    return result;
  }
  setMutationDefaults(mutationKey, options) {
    __privateGet(this, _mutationDefaults).set(hashKey(mutationKey), {
      mutationKey,
      defaultOptions: options
    });
  }
  getMutationDefaults(mutationKey) {
    const defaults = [...__privateGet(this, _mutationDefaults).values()];
    const result = {};
    defaults.forEach((queryDefault) => {
      if (partialMatchKey(mutationKey, queryDefault.mutationKey)) {
        Object.assign(result, queryDefault.defaultOptions);
      }
    });
    return result;
  }
  defaultQueryOptions(options) {
    if (options._defaulted) {
      return options;
    }
    const defaultedOptions = {
      ...__privateGet(this, _defaultOptions2).queries,
      ...this.getQueryDefaults(options.queryKey),
      ...options,
      _defaulted: true
    };
    if (!defaultedOptions.queryHash) {
      defaultedOptions.queryHash = hashQueryKeyByOptions(
        defaultedOptions.queryKey,
        defaultedOptions
      );
    }
    if (defaultedOptions.refetchOnReconnect === void 0) {
      defaultedOptions.refetchOnReconnect = defaultedOptions.networkMode !== "always";
    }
    if (defaultedOptions.throwOnError === void 0) {
      defaultedOptions.throwOnError = !!defaultedOptions.suspense;
    }
    if (!defaultedOptions.networkMode && defaultedOptions.persister) {
      defaultedOptions.networkMode = "offlineFirst";
    }
    if (defaultedOptions.queryFn === skipToken) {
      defaultedOptions.enabled = false;
    }
    return defaultedOptions;
  }
  defaultMutationOptions(options) {
    if (options == null ? void 0 : options._defaulted) {
      return options;
    }
    return {
      ...__privateGet(this, _defaultOptions2).mutations,
      ...(options == null ? void 0 : options.mutationKey) && this.getMutationDefaults(options.mutationKey),
      ...options,
      _defaulted: true
    };
  }
  clear() {
    __privateGet(this, _queryCache).clear();
    __privateGet(this, _mutationCache2).clear();
  }
}, _queryCache = new WeakMap(), _mutationCache2 = new WeakMap(), _defaultOptions2 = new WeakMap(), _queryDefaults = new WeakMap(), _mutationDefaults = new WeakMap(), _mountCount = new WeakMap(), _unsubscribeFocus = new WeakMap(), _unsubscribeOnline = new WeakMap(), _k);
var QueryClientContext = reactExports.createContext(
  void 0
);
var useQueryClient = (queryClient2) => {
  const client2 = reactExports.useContext(QueryClientContext);
  if (queryClient2) {
    return queryClient2;
  }
  if (!client2) {
    throw new Error("No QueryClient set, use QueryClientProvider to set one");
  }
  return client2;
};
var QueryClientProvider = ({
  client: client2,
  children
}) => {
  reactExports.useEffect(() => {
    client2.mount();
    return () => {
      client2.unmount();
    };
  }, [client2]);
  return /* @__PURE__ */ jsxRuntimeExports.jsx(QueryClientContext.Provider, { value: client2, children });
};
var IsRestoringContext = reactExports.createContext(false);
var useIsRestoring = () => reactExports.useContext(IsRestoringContext);
IsRestoringContext.Provider;
function createValue() {
  let isReset = false;
  return {
    clearReset: () => {
      isReset = false;
    },
    reset: () => {
      isReset = true;
    },
    isReset: () => {
      return isReset;
    }
  };
}
var QueryErrorResetBoundaryContext = reactExports.createContext(createValue());
var useQueryErrorResetBoundary = () => reactExports.useContext(QueryErrorResetBoundaryContext);
var ensurePreventErrorBoundaryRetry = (options, errorResetBoundary) => {
  if (options.suspense || options.throwOnError || options.experimental_prefetchInRender) {
    if (!errorResetBoundary.isReset()) {
      options.retryOnMount = false;
    }
  }
};
var useClearResetErrorBoundary = (errorResetBoundary) => {
  reactExports.useEffect(() => {
    errorResetBoundary.clearReset();
  }, [errorResetBoundary]);
};
var getHasError = ({
  result,
  errorResetBoundary,
  throwOnError,
  query,
  suspense
}) => {
  return result.isError && !errorResetBoundary.isReset() && !result.isFetching && query && (suspense && result.data === void 0 || shouldThrowError(throwOnError, [result.error, query]));
};
var ensureSuspenseTimers = (defaultedOptions) => {
  if (defaultedOptions.suspense) {
    const MIN_SUSPENSE_TIME_MS = 1e3;
    const clamp = (value) => value === "static" ? value : Math.max(value ?? MIN_SUSPENSE_TIME_MS, MIN_SUSPENSE_TIME_MS);
    const originalStaleTime = defaultedOptions.staleTime;
    defaultedOptions.staleTime = typeof originalStaleTime === "function" ? (...args) => clamp(originalStaleTime(...args)) : clamp(originalStaleTime);
    if (typeof defaultedOptions.gcTime === "number") {
      defaultedOptions.gcTime = Math.max(
        defaultedOptions.gcTime,
        MIN_SUSPENSE_TIME_MS
      );
    }
  }
};
var willFetch = (result, isRestoring) => result.isLoading && result.isFetching && !isRestoring;
var shouldSuspend = (defaultedOptions, result) => (defaultedOptions == null ? void 0 : defaultedOptions.suspense) && result.isPending;
var fetchOptimistic = (defaultedOptions, observer, errorResetBoundary) => observer.fetchOptimistic(defaultedOptions).catch(() => {
  errorResetBoundary.clearReset();
});
function useBaseQuery(options, Observer, queryClient2) {
  var _a2, _b2, _c2, _d2, _e2;
  const isRestoring = useIsRestoring();
  const errorResetBoundary = useQueryErrorResetBoundary();
  const client2 = useQueryClient(queryClient2);
  const defaultedOptions = client2.defaultQueryOptions(options);
  (_b2 = (_a2 = client2.getDefaultOptions().queries) == null ? void 0 : _a2._experimental_beforeQuery) == null ? void 0 : _b2.call(
    _a2,
    defaultedOptions
  );
  defaultedOptions._optimisticResults = isRestoring ? "isRestoring" : "optimistic";
  ensureSuspenseTimers(defaultedOptions);
  ensurePreventErrorBoundaryRetry(defaultedOptions, errorResetBoundary);
  useClearResetErrorBoundary(errorResetBoundary);
  const isNewCacheEntry = !client2.getQueryCache().get(defaultedOptions.queryHash);
  const [observer] = reactExports.useState(
    () => new Observer(
      client2,
      defaultedOptions
    )
  );
  const result = observer.getOptimisticResult(defaultedOptions);
  const shouldSubscribe = !isRestoring && options.subscribed !== false;
  reactExports.useSyncExternalStore(
    reactExports.useCallback(
      (onStoreChange) => {
        const unsubscribe = shouldSubscribe ? observer.subscribe(notifyManager.batchCalls(onStoreChange)) : noop;
        observer.updateResult();
        return unsubscribe;
      },
      [observer, shouldSubscribe]
    ),
    () => observer.getCurrentResult(),
    () => observer.getCurrentResult()
  );
  reactExports.useEffect(() => {
    observer.setOptions(defaultedOptions);
  }, [defaultedOptions, observer]);
  if (shouldSuspend(defaultedOptions, result)) {
    throw fetchOptimistic(defaultedOptions, observer, errorResetBoundary);
  }
  if (getHasError({
    result,
    errorResetBoundary,
    throwOnError: defaultedOptions.throwOnError,
    query: client2.getQueryCache().get(defaultedOptions.queryHash),
    suspense: defaultedOptions.suspense
  })) {
    throw result.error;
  }
  (_d2 = (_c2 = client2.getDefaultOptions().queries) == null ? void 0 : _c2._experimental_afterQuery) == null ? void 0 : _d2.call(
    _c2,
    defaultedOptions,
    result
  );
  if (defaultedOptions.experimental_prefetchInRender && !isServer && willFetch(result, isRestoring)) {
    const promise = isNewCacheEntry ? (
      // Fetch immediately on render in order to ensure `.promise` is resolved even if the component is unmounted
      fetchOptimistic(defaultedOptions, observer, errorResetBoundary)
    ) : (
      // subscribe to the "cache promise" so that we can finalize the currentThenable once data comes in
      (_e2 = client2.getQueryCache().get(defaultedOptions.queryHash)) == null ? void 0 : _e2.promise
    );
    promise == null ? void 0 : promise.catch(noop).finally(() => {
      observer.updateResult();
    });
  }
  return !defaultedOptions.notifyOnChangeProps ? observer.trackResult(result) : result;
}
function useQuery(options, queryClient2) {
  return useBaseQuery(options, QueryObserver, queryClient2);
}
function useMutation(options, queryClient2) {
  const client2 = useQueryClient(queryClient2);
  const [observer] = reactExports.useState(
    () => new MutationObserver$1(
      client2,
      options
    )
  );
  reactExports.useEffect(() => {
    observer.setOptions(options);
  }, [observer, options]);
  const result = reactExports.useSyncExternalStore(
    reactExports.useCallback(
      (onStoreChange) => observer.subscribe(notifyManager.batchCalls(onStoreChange)),
      [observer]
    ),
    () => observer.getCurrentResult(),
    () => observer.getCurrentResult()
  );
  const mutate = reactExports.useCallback(
    (variables, mutateOptions) => {
      observer.mutate(variables, mutateOptions).catch(noop);
    },
    [observer]
  );
  if (result.error && shouldThrowError(observer.options.throwOnError, [result.error])) {
    throw result.error;
  }
  return { ...result, mutate, mutateAsync: result.mutate };
}
/**
 * @license lucide-react v0.294.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
var defaultAttributes = {
  xmlns: "http://www.w3.org/2000/svg",
  width: 24,
  height: 24,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 2,
  strokeLinecap: "round",
  strokeLinejoin: "round"
};
/**
 * @license lucide-react v0.294.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const toKebabCase = (string) => string.replace(/([a-z0-9])([A-Z])/g, "$1-$2").toLowerCase().trim();
const createLucideIcon = (iconName, iconNode) => {
  const Component = reactExports.forwardRef(
    ({ color = "currentColor", size = 24, strokeWidth = 2, absoluteStrokeWidth, className = "", children, ...rest }, ref) => reactExports.createElement(
      "svg",
      {
        ref,
        ...defaultAttributes,
        width: size,
        height: size,
        stroke: color,
        strokeWidth: absoluteStrokeWidth ? Number(strokeWidth) * 24 / Number(size) : strokeWidth,
        className: ["lucide", `lucide-${toKebabCase(iconName)}`, className].join(" "),
        ...rest
      },
      [
        ...iconNode.map(([tag, attrs]) => reactExports.createElement(tag, attrs)),
        ...Array.isArray(children) ? children : [children]
      ]
    )
  );
  Component.displayName = `${iconName}`;
  return Component;
};
/**
 * @license lucide-react v0.294.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const AlertCircle = createLucideIcon("AlertCircle", [
  ["circle", { cx: "12", cy: "12", r: "10", key: "1mglay" }],
  ["line", { x1: "12", x2: "12", y1: "8", y2: "12", key: "1pkeuh" }],
  ["line", { x1: "12", x2: "12.01", y1: "16", y2: "16", key: "4dfq90" }]
]);
/**
 * @license lucide-react v0.294.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const AlertTriangle = createLucideIcon("AlertTriangle", [
  [
    "path",
    {
      d: "m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z",
      key: "c3ski4"
    }
  ],
  ["path", { d: "M12 9v4", key: "juzpu7" }],
  ["path", { d: "M12 17h.01", key: "p32p05" }]
]);
/**
 * @license lucide-react v0.294.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const CheckCircle = createLucideIcon("CheckCircle", [
  ["path", { d: "M22 11.08V12a10 10 0 1 1-5.93-9.14", key: "g774vq" }],
  ["path", { d: "m9 11 3 3L22 4", key: "1pflzl" }]
]);
/**
 * @license lucide-react v0.294.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const ChevronDown = createLucideIcon("ChevronDown", [
  ["path", { d: "m6 9 6 6 6-6", key: "qrunsl" }]
]);
/**
 * @license lucide-react v0.294.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const ChevronUp = createLucideIcon("ChevronUp", [["path", { d: "m18 15-6-6-6 6", key: "153udz" }]]);
/**
 * @license lucide-react v0.294.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const ChevronsDown = createLucideIcon("ChevronsDown", [
  ["path", { d: "m7 6 5 5 5-5", key: "1lc07p" }],
  ["path", { d: "m7 13 5 5 5-5", key: "1d48rs" }]
]);
/**
 * @license lucide-react v0.294.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const ChevronsUp = createLucideIcon("ChevronsUp", [
  ["path", { d: "m17 11-5-5-5 5", key: "e8nh98" }],
  ["path", { d: "m17 18-5-5-5 5", key: "2avn1x" }]
]);
/**
 * @license lucide-react v0.294.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const Clock = createLucideIcon("Clock", [
  ["circle", { cx: "12", cy: "12", r: "10", key: "1mglay" }],
  ["polyline", { points: "12 6 12 12 16 14", key: "68esgv" }]
]);
/**
 * @license lucide-react v0.294.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const Copy = createLucideIcon("Copy", [
  ["rect", { width: "14", height: "14", x: "8", y: "8", rx: "2", ry: "2", key: "17jyea" }],
  ["path", { d: "M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2", key: "zix9uf" }]
]);
/**
 * @license lucide-react v0.294.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const Cpu = createLucideIcon("Cpu", [
  ["rect", { x: "4", y: "4", width: "16", height: "16", rx: "2", key: "1vbyd7" }],
  ["rect", { x: "9", y: "9", width: "6", height: "6", key: "o3kz5p" }],
  ["path", { d: "M15 2v2", key: "13l42r" }],
  ["path", { d: "M15 20v2", key: "15mkzm" }],
  ["path", { d: "M2 15h2", key: "1gxd5l" }],
  ["path", { d: "M2 9h2", key: "1bbxkp" }],
  ["path", { d: "M20 15h2", key: "19e6y8" }],
  ["path", { d: "M20 9h2", key: "19tzq7" }],
  ["path", { d: "M9 2v2", key: "165o2o" }],
  ["path", { d: "M9 20v2", key: "i2bqo8" }]
]);
/**
 * @license lucide-react v0.294.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const Info = createLucideIcon("Info", [
  ["circle", { cx: "12", cy: "12", r: "10", key: "1mglay" }],
  ["path", { d: "M12 16v-4", key: "1dtifu" }],
  ["path", { d: "M12 8h.01", key: "e9boi3" }]
]);
/**
 * @license lucide-react v0.294.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const Minus = createLucideIcon("Minus", [["path", { d: "M5 12h14", key: "1ays0h" }]]);
/**
 * @license lucide-react v0.294.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const Pen = createLucideIcon("Pen", [
  ["path", { d: "M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z", key: "5qss01" }]
]);
/**
 * @license lucide-react v0.294.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const Plus = createLucideIcon("Plus", [
  ["path", { d: "M5 12h14", key: "1ays0h" }],
  ["path", { d: "M12 5v14", key: "s699le" }]
]);
/**
 * @license lucide-react v0.294.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const Radio = createLucideIcon("Radio", [
  ["path", { d: "M4.9 19.1C1 15.2 1 8.8 4.9 4.9", key: "1vaf9d" }],
  ["path", { d: "M7.8 16.2c-2.3-2.3-2.3-6.1 0-8.5", key: "u1ii0m" }],
  ["circle", { cx: "12", cy: "12", r: "2", key: "1c9p78" }],
  ["path", { d: "M16.2 7.8c2.3 2.3 2.3 6.1 0 8.5", key: "1j5fej" }],
  ["path", { d: "M19.1 4.9C23 8.8 23 15.1 19.1 19", key: "10b0cb" }]
]);
/**
 * @license lucide-react v0.294.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const Save = createLucideIcon("Save", [
  ["path", { d: "M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z", key: "1owoqh" }],
  ["polyline", { points: "17 21 17 13 7 13 7 21", key: "1md35c" }],
  ["polyline", { points: "7 3 7 8 15 8", key: "8nz8an" }]
]);
/**
 * @license lucide-react v0.294.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const Settings = createLucideIcon("Settings", [
  [
    "path",
    {
      d: "M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z",
      key: "1qme2f"
    }
  ],
  ["circle", { cx: "12", cy: "12", r: "3", key: "1v7zrd" }]
]);
/**
 * @license lucide-react v0.294.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const Star = createLucideIcon("Star", [
  [
    "polygon",
    {
      points: "12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2",
      key: "8f66p6"
    }
  ]
]);
/**
 * @license lucide-react v0.294.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const Trash2 = createLucideIcon("Trash2", [
  ["path", { d: "M3 6h18", key: "d0wm0j" }],
  ["path", { d: "M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6", key: "4alrt4" }],
  ["path", { d: "M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2", key: "v07s0e" }],
  ["line", { x1: "10", x2: "10", y1: "11", y2: "17", key: "1uufr5" }],
  ["line", { x1: "14", x2: "14", y1: "11", y2: "17", key: "xtxkd" }]
]);
/**
 * @license lucide-react v0.294.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const Volume2 = createLucideIcon("Volume2", [
  ["polygon", { points: "11 5 6 9 2 9 2 15 6 15 11 19 11 5", key: "16drj5" }],
  ["path", { d: "M15.54 8.46a5 5 0 0 1 0 7.07", key: "ltjumu" }],
  ["path", { d: "M19.07 4.93a10 10 0 0 1 0 14.14", key: "1kegas" }]
]);
/**
 * @license lucide-react v0.294.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const VolumeX = createLucideIcon("VolumeX", [
  ["polygon", { points: "11 5 6 9 2 9 2 15 6 15 11 19 11 5", key: "16drj5" }],
  ["line", { x1: "22", x2: "16", y1: "9", y2: "15", key: "1ewh16" }],
  ["line", { x1: "16", x2: "22", y1: "9", y2: "15", key: "5ykzw1" }]
]);
/**
 * @license lucide-react v0.294.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const Wand2 = createLucideIcon("Wand2", [
  [
    "path",
    {
      d: "m21.64 3.64-1.28-1.28a1.21 1.21 0 0 0-1.72 0L2.36 18.64a1.21 1.21 0 0 0 0 1.72l1.28 1.28a1.2 1.2 0 0 0 1.72 0L21.64 5.36a1.2 1.2 0 0 0 0-1.72Z",
      key: "1bcowg"
    }
  ],
  ["path", { d: "m14 7 3 3", key: "1r5n42" }],
  ["path", { d: "M5 6v4", key: "ilb8ba" }],
  ["path", { d: "M19 14v4", key: "blhpug" }],
  ["path", { d: "M10 2v2", key: "7u0qdc" }],
  ["path", { d: "M7 8H3", key: "zfb6yr" }],
  ["path", { d: "M21 16h-4", key: "1cnmox" }],
  ["path", { d: "M11 3H9", key: "1obp7u" }]
]);
/**
 * @license lucide-react v0.294.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const X = createLucideIcon("X", [
  ["path", { d: "M18 6 6 18", key: "1bl5f8" }],
  ["path", { d: "m6 6 12 12", key: "d8bk6v" }]
]);
function r(e) {
  var t2, f2, n2 = "";
  if ("string" == typeof e || "number" == typeof e)
    n2 += e;
  else if ("object" == typeof e)
    if (Array.isArray(e)) {
      var o = e.length;
      for (t2 = 0; t2 < o; t2++)
        e[t2] && (f2 = r(e[t2])) && (n2 && (n2 += " "), n2 += f2);
    } else
      for (f2 in e)
        e[f2] && (n2 && (n2 += " "), n2 += f2);
  return n2;
}
function clsx() {
  for (var e, t2, f2 = 0, n2 = "", o = arguments.length; f2 < o; f2++)
    (e = arguments[f2]) && (t2 = r(e)) && (n2 && (n2 += " "), n2 += t2);
  return n2;
}
const BREAKPOINTS = ["sm", "md", "lg", "xl", "xxl"];
const toArray = (value) => {
  if (typeof value === "string") {
    return [value];
  }
  return Array.isArray(value) ? value : [];
};
const isResponsiveMap = (value) => {
  return typeof value === "object" && value !== null;
};
const resolveResponsiveProp = (propValue, generator) => {
  if (propValue == null) {
    return [];
  }
  if (!isResponsiveMap(propValue)) {
    return toArray(generator(propValue));
  }
  const classNames = [];
  if (propValue.base != null) {
    classNames.push(...toArray(generator(propValue.base)));
  }
  for (const breakpoint of BREAKPOINTS) {
    const value = propValue[breakpoint];
    if (value != null) {
      classNames.push(...toArray(generator(value, breakpoint)));
    }
  }
  return classNames;
};
const resolveDirectionClass = (direction, breakpoint) => {
  const prefix = breakpoint ? `flex-${breakpoint}` : "flex";
  switch (direction) {
    case "row":
      return `${prefix}-row`;
    case "row-reverse":
      return `${prefix}-row-reverse`;
    case "column":
      return `${prefix}-column`;
    case "column-reverse":
      return `${prefix}-column-reverse`;
    default:
      return void 0;
  }
};
const resolveWrapClass = (wrap, breakpoint) => {
  const prefix = breakpoint ? `flex-${breakpoint}` : "flex";
  switch (wrap) {
    case "wrap":
      return `${prefix}-wrap`;
    case "wrap-reverse":
      return `${prefix}-wrap-reverse`;
    case "nowrap":
      return `${prefix}-nowrap`;
    default:
      return void 0;
  }
};
const resolveJustifyClass = (justify, breakpoint) => {
  const prefix = breakpoint ? `justify-content-${breakpoint}` : "justify-content";
  return `${prefix}-${justify}`;
};
const resolveAlignItemsClass = (align, breakpoint) => {
  const prefix = breakpoint ? `align-items-${breakpoint}` : "align-items";
  return `${prefix}-${align}`;
};
const resolveAlignContentClass = (align, breakpoint) => {
  const prefix = breakpoint ? `align-content-${breakpoint}` : "align-content";
  return `${prefix}-${align}`;
};
const resolveGapClass = (gap, breakpoint) => {
  const prefix = breakpoint ? `gap-${breakpoint}` : "gap";
  return `${prefix}-${gap}`;
};
const Flex = reactExports.forwardRef(
  ({
    as,
    inline = false,
    direction,
    wrap,
    justify,
    align,
    alignContent,
    gap,
    className,
    ...rest
  }, forwardedRef) => {
    const ref = forwardedRef;
    const Component = as ?? "div";
    const directionClasses = resolveResponsiveProp(direction, resolveDirectionClass);
    const wrapClasses = resolveResponsiveProp(wrap, resolveWrapClass);
    const justifyClasses = resolveResponsiveProp(justify, resolveJustifyClass);
    const alignClasses = resolveResponsiveProp(align, resolveAlignItemsClass);
    const alignContentClasses = resolveResponsiveProp(alignContent, resolveAlignContentClass);
    const gapClasses = resolveResponsiveProp(gap, resolveGapClass);
    return /* @__PURE__ */ jsxRuntimeExports.jsx(
      Component,
      {
        ...rest,
        ref,
        className: clsx(
          inline ? "d-inline-flex" : "d-flex",
          directionClasses,
          wrapClasses,
          justifyClasses,
          alignClasses,
          alignContentClasses,
          gapClasses,
          className
        )
      }
    );
  }
);
Flex.displayName = "Flex";
function Toast({ id: id2, type, message, duration = 3e3, onClose }) {
  reactExports.useEffect(() => {
    if (duration > 0) {
      const timer = setTimeout(() => {
        onClose(id2);
      }, duration);
      return () => clearTimeout(timer);
    }
  }, [id2, duration, onClose]);
  const getIcon = () => {
    switch (type) {
      case "success":
        return /* @__PURE__ */ jsxRuntimeExports.jsx(CheckCircle, { size: 20 });
      case "error":
        return /* @__PURE__ */ jsxRuntimeExports.jsx(AlertCircle, { size: 20 });
      case "warning":
        return /* @__PURE__ */ jsxRuntimeExports.jsx(AlertTriangle, { size: 20 });
      case "info":
      default:
        return /* @__PURE__ */ jsxRuntimeExports.jsx(Info, { size: 20 });
    }
  };
  const getColorClass = () => {
    switch (type) {
      case "success":
        return "toast-success";
      case "error":
        return "toast-error";
      case "warning":
        return "toast-warning";
      case "info":
      default:
        return "toast-info";
    }
  };
  return /* @__PURE__ */ jsxRuntimeExports.jsx(
    "div",
    {
      className: `toast-item ${getColorClass()}`,
      role: "alert",
      "aria-live": "polite",
      "aria-atomic": "true",
      children: /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { align: "start", gap: 2, children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "toast-icon", children: getIcon() }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "toast-message", children: message }),
        /* @__PURE__ */ jsxRuntimeExports.jsx(
          "button",
          {
            className: "toast-close",
            onClick: () => onClose(id2),
            "aria-label": "Close notification",
            children: /* @__PURE__ */ jsxRuntimeExports.jsx(X, { size: 16 })
          }
        )
      ] })
    }
  );
}
const ToastContext = reactExports.createContext(void 0);
function ToastProvider({ children }) {
  const [toasts, setToasts] = reactExports.useState([]);
  const removeToast = reactExports.useCallback((id2) => {
    setToasts((prev) => prev.filter((t2) => t2.id !== id2));
  }, []);
  const showToast = reactExports.useCallback((type, message, duration) => {
    const id2 = `toast-${Date.now()}-${Math.random()}`;
    setToasts((prev) => [...prev, { id: id2, type, message, duration }]);
  }, []);
  const success = reactExports.useCallback((message, duration) => {
    showToast("success", message, duration);
  }, [showToast]);
  const error = reactExports.useCallback((message, duration) => {
    showToast("error", message, duration);
  }, [showToast]);
  const info = reactExports.useCallback((message, duration) => {
    showToast("info", message, duration);
  }, [showToast]);
  const warning = reactExports.useCallback((message, duration) => {
    showToast("warning", message, duration);
  }, [showToast]);
  return /* @__PURE__ */ jsxRuntimeExports.jsxs(ToastContext.Provider, { value: { showToast, success, error, info, warning }, children: [
    children,
    /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "toast-container", children: toasts.map((toast) => /* @__PURE__ */ jsxRuntimeExports.jsx(
      Toast,
      {
        id: toast.id,
        type: toast.type,
        message: toast.message,
        duration: toast.duration,
        onClose: removeToast
      },
      toast.id
    )) })
  ] });
}
function useToast() {
  const context = reactExports.useContext(ToastContext);
  if (!context) {
    throw new Error("useToast must be used within ToastProvider");
  }
  return context;
}
async function fetchDevices() {
  const response = await fetch("/api/v1/devices");
  if (!response.ok) {
    throw new Error("Failed to fetch devices");
  }
  return response.json();
}
function useDevices() {
  return useQuery({
    queryKey: ["devices"],
    queryFn: fetchDevices,
    staleTime: 3e4,
    // Cache for 30 seconds
    refetchInterval: 6e4
    // Refetch every minute
  });
}
async function fetchCaptures() {
  const response = await fetch("/api/v1/captures");
  if (!response.ok) {
    throw new Error("Failed to fetch captures");
  }
  return response.json();
}
async function updateCapture(captureId, request) {
  const response = await fetch(`/api/v1/captures/${captureId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request)
  });
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Failed to update capture");
  }
  return response.json();
}
async function createCapture(request) {
  const response = await fetch("/api/v1/captures", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request)
  });
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Failed to create capture");
  }
  return response.json();
}
async function startCapture(captureId) {
  const response = await fetch(`/api/v1/captures/${captureId}/start`, {
    method: "POST"
  });
  if (!response.ok) {
    throw new Error("Failed to start capture");
  }
  return response.json();
}
async function stopCapture(captureId) {
  const response = await fetch(`/api/v1/captures/${captureId}/stop`, {
    method: "POST"
  });
  if (!response.ok) {
    throw new Error("Failed to stop capture");
  }
  return response.json();
}
async function deleteCapture(captureId) {
  const response = await fetch(`/api/v1/captures/${captureId}`, {
    method: "DELETE"
  });
  if (!response.ok) {
    throw new Error("Failed to delete capture");
  }
}
function useCaptures() {
  return useQuery({
    queryKey: ["captures"],
    queryFn: fetchCaptures,
    refetchInterval: 2e3
    // Poll every 2 seconds to catch state changes
  });
}
function useUpdateCapture() {
  const queryClient2 = useQueryClient();
  return useMutation({
    mutationFn: ({ captureId, request }) => updateCapture(captureId, request),
    onSuccess: () => {
      queryClient2.invalidateQueries({ queryKey: ["captures"] });
    }
  });
}
function useCreateCapture() {
  const queryClient2 = useQueryClient();
  return useMutation({
    mutationFn: (request) => createCapture(request),
    onSuccess: () => {
      queryClient2.invalidateQueries({ queryKey: ["captures"] });
    }
  });
}
function useStartCapture() {
  const queryClient2 = useQueryClient();
  return useMutation({
    mutationFn: (captureId) => startCapture(captureId),
    onSuccess: () => {
      queryClient2.invalidateQueries({ queryKey: ["captures"] });
    }
  });
}
function useStopCapture() {
  const queryClient2 = useQueryClient();
  return useMutation({
    mutationFn: (captureId) => stopCapture(captureId),
    onSuccess: () => {
      queryClient2.invalidateQueries({ queryKey: ["captures"] });
    }
  });
}
function useDeleteCapture() {
  const queryClient2 = useQueryClient();
  return useMutation({
    mutationFn: (captureId) => deleteCapture(captureId),
    onSuccess: () => {
      queryClient2.invalidateQueries({ queryKey: ["captures"] });
      queryClient2.invalidateQueries({ queryKey: ["channels"] });
    }
  });
}
async function fetchChannels(captureId) {
  const response = await fetch(`/api/v1/captures/${captureId}/channels`);
  if (!response.ok) {
    throw new Error("Failed to fetch channels");
  }
  return response.json();
}
async function createChannel(captureId, request) {
  const response = await fetch(`/api/v1/captures/${captureId}/channels`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request)
  });
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Failed to create channel");
  }
  return response.json();
}
async function deleteChannel(channelId) {
  const response = await fetch(`/api/v1/channels/${channelId}`, {
    method: "DELETE"
  });
  if (!response.ok) {
    throw new Error("Failed to delete channel");
  }
}
async function updateChannel(channelId, request) {
  const response = await fetch(`/api/v1/channels/${channelId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request)
  });
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Failed to update channel");
  }
  return response.json();
}
async function startChannel(channelId) {
  const response = await fetch(`/api/v1/channels/${channelId}/start`, {
    method: "POST"
  });
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Failed to start channel");
  }
  return response.json();
}
async function stopChannel(channelId) {
  const response = await fetch(`/api/v1/channels/${channelId}/stop`, {
    method: "POST"
  });
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Failed to stop channel");
  }
  return response.json();
}
function useChannels(captureId) {
  return useQuery({
    queryKey: ["channels", captureId],
    queryFn: () => fetchChannels(captureId),
    enabled: !!captureId,
    refetchInterval: 2e3
    // Poll every 2 seconds
  });
}
function useCreateChannel() {
  const queryClient2 = useQueryClient();
  return useMutation({
    mutationFn: ({ captureId, request }) => createChannel(captureId, request),
    onSuccess: (_, variables) => {
      queryClient2.invalidateQueries({ queryKey: ["channels", variables.captureId] });
    }
  });
}
function useDeleteChannel() {
  const queryClient2 = useQueryClient();
  return useMutation({
    mutationFn: (channelId) => deleteChannel(channelId),
    onSuccess: () => {
      queryClient2.invalidateQueries({ queryKey: ["channels"] });
    }
  });
}
function useUpdateChannel(captureId) {
  const queryClient2 = useQueryClient();
  return useMutation({
    mutationFn: ({ channelId, request }) => updateChannel(channelId, request),
    onSuccess: () => {
      queryClient2.invalidateQueries({ queryKey: ["channels", captureId] });
    }
  });
}
function useStartChannel(captureId) {
  const queryClient2 = useQueryClient();
  return useMutation({
    mutationFn: (channelId) => startChannel(channelId),
    onSuccess: () => {
      queryClient2.invalidateQueries({ queryKey: ["channels", captureId] });
    }
  });
}
function useStopChannel(captureId) {
  const queryClient2 = useQueryClient();
  return useMutation({
    mutationFn: (channelId) => stopChannel(channelId),
    onSuccess: () => {
      queryClient2.invalidateQueries({ queryKey: ["channels", captureId] });
    }
  });
}
const MEMORY_BANKS_KEY = "wavecapsdr_memory_banks";
const MAX_MEMORY_BANKS = 50;
function useMemoryBanks() {
  const [memoryBanks, setMemoryBanks] = reactExports.useState([]);
  reactExports.useEffect(() => {
    try {
      const stored = localStorage.getItem(MEMORY_BANKS_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        setMemoryBanks(parsed);
      }
    } catch (error) {
      console.error("Failed to load memory banks:", error);
    }
  }, []);
  const saveMemoryBanks = reactExports.useCallback((banks) => {
    try {
      localStorage.setItem(MEMORY_BANKS_KEY, JSON.stringify(banks));
      setMemoryBanks(banks);
    } catch (error) {
      console.error("Failed to save memory banks:", error);
    }
  }, []);
  const saveToMemoryBank = reactExports.useCallback((name, capture, channels) => {
    const newBank = {
      id: `mb_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`,
      name,
      timestamp: Date.now(),
      captureConfig: {
        centerHz: capture.centerHz,
        sampleRate: capture.sampleRate,
        gain: capture.gain,
        bandwidth: capture.bandwidth,
        ppm: capture.ppm,
        antenna: capture.antenna,
        deviceId: capture.deviceId
      },
      channels: channels.map((ch2) => ({
        mode: ch2.mode,
        offsetHz: ch2.offsetHz,
        audioRate: ch2.audioRate,
        squelchDb: ch2.squelchDb,
        name: ch2.name
      }))
    };
    const newBanks = [newBank, ...memoryBanks].slice(0, MAX_MEMORY_BANKS);
    saveMemoryBanks(newBanks);
    return newBank;
  }, [memoryBanks, saveMemoryBanks]);
  const deleteMemoryBank = reactExports.useCallback((id2) => {
    const newBanks = memoryBanks.filter((bank) => bank.id !== id2);
    saveMemoryBanks(newBanks);
  }, [memoryBanks, saveMemoryBanks]);
  const renameMemoryBank = reactExports.useCallback((id2, newName) => {
    const newBanks = memoryBanks.map(
      (bank) => bank.id === id2 ? { ...bank, name: newName } : bank
    );
    saveMemoryBanks(newBanks);
  }, [memoryBanks, saveMemoryBanks]);
  const getMemoryBank = reactExports.useCallback((id2) => {
    return memoryBanks.find((bank) => bank.id === id2);
  }, [memoryBanks]);
  const clearMemoryBanks = reactExports.useCallback(() => {
    try {
      localStorage.removeItem(MEMORY_BANKS_KEY);
      setMemoryBanks([]);
    } catch (error) {
      console.error("Failed to clear memory banks:", error);
    }
  }, []);
  return {
    memoryBanks,
    saveToMemoryBank,
    deleteMemoryBank,
    renameMemoryBank,
    getMemoryBank,
    clearMemoryBanks
  };
}
function useDebounce(value, delay) {
  const [debouncedValue, setDebouncedValue] = reactExports.useState(value);
  reactExports.useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);
    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);
  return debouncedValue;
}
function formatFrequencyMHz(hz) {
  const mhz = hz / 1e6;
  return mhz.toFixed(3);
}
function formatSampleRate(hz) {
  if (hz >= 1e6) {
    return `${(hz / 1e6).toFixed(2)} MHz`;
  } else if (hz >= 1e3) {
    return `${(hz / 1e3).toFixed(0)} kHz`;
  }
  return `${hz} Hz`;
}
function getDeviceDisplayName(device) {
  if (device.nickname) {
    return device.nickname;
  }
  if (device.shorthand) {
    return device.shorthand;
  }
  return device.label;
}
const BUTTON_STYLES = {
  default: { filled: "btn-secondary", outline: "btn-outline-secondary" },
  primary: { filled: "btn-primary", outline: "btn-outline-primary" },
  secondary: { filled: "btn-secondary", outline: "btn-outline-secondary" },
  success: { filled: "btn-success", outline: "btn-outline-success" },
  create: { filled: "btn-success", outline: "btn-outline-success" },
  danger: { filled: "btn-danger", outline: "btn-outline-danger" },
  destroy: { filled: "btn-danger", outline: "btn-outline-danger" },
  warning: { filled: "btn-warning", outline: "btn-outline-warning" },
  light: { filled: "btn-light", outline: "btn-outline-light" },
  link: { filled: "btn-link" },
  close: { filled: "btn-close", includeBaseClass: false },
  unstyled: { filled: "", includeBaseClass: false }
};
const Button = reactExports.forwardRef(
  ({
    use = "default",
    appearance = "filled",
    size = "md",
    startContent,
    endContent,
    tooltip,
    className,
    children,
    type = "button",
    isContentInline,
    isCondensed = false,
    title: htmlTitle,
    ...rest
  }, ref) => {
    const { ["aria-label"]: ariaLabelProp, ...buttonRest } = rest;
    const style = BUTTON_STYLES[use] ?? BUTTON_STYLES.default;
    const includeBaseClass = style.includeBaseClass ?? true;
    const variantClass = appearance === "outline" ? style.outline ?? style.filled ?? "" : style.filled ?? style.outline ?? "";
    const shouldApplyContentLayout = isContentInline ?? Boolean(startContent || endContent);
    const sizeClass = includeBaseClass && size !== "md" ? `btn-${size}` : void 0;
    const labelText = typeof children === "string" ? children : typeof tooltip === "string" ? tooltip : typeof htmlTitle === "string" ? htmlTitle : void 0;
    const resolvedTitle = tooltip ?? htmlTitle ?? (isCondensed ? labelText : void 0);
    const resolvedAriaLabel = ariaLabelProp ?? (isCondensed ? labelText : void 0);
    return /* @__PURE__ */ jsxRuntimeExports.jsxs(
      "button",
      {
        ...buttonRest,
        ref,
        type,
        title: resolvedTitle,
        "aria-label": resolvedAriaLabel,
        className: clsx(
          includeBaseClass && "btn",
          includeBaseClass && sizeClass,
          variantClass,
          includeBaseClass && shouldApplyContentLayout && "d-inline-flex align-items-center gap-2",
          isCondensed && "btn-condensed",
          className
        ),
        children: [
          startContent,
          isCondensed ? children != null && /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "visually-hidden", children }) : children,
          endContent
        ]
      }
    );
  }
);
Button.displayName = "Button";
const Slider = reactExports.forwardRef(
  ({
    label,
    value,
    min,
    max,
    step = 1,
    coarseStep,
    unit = "",
    formatValue,
    parseValue,
    showMinMax = true,
    info,
    onChange,
    className,
    disabled,
    ...rest
  }, ref) => {
    const [inputValue, setInputValue] = reactExports.useState("");
    const [isEditing, setIsEditing] = reactExports.useState(false);
    const displayValue = formatValue ? formatValue(value) : value.toLocaleString();
    const actualCoarseStep = coarseStep ?? step * 10;
    const clamp = (val) => Math.max(min, Math.min(max, val));
    const handleIncrement = (amount) => {
      onChange(clamp(value + amount));
    };
    const handleTextChange = (text) => {
      setInputValue(text);
    };
    const handleTextBlur = () => {
      if (inputValue) {
        let newValue = null;
        if (parseValue) {
          newValue = parseValue(inputValue);
        } else {
          const parsed = parseFloat(inputValue.replace(/,/g, ""));
          if (!isNaN(parsed)) {
            newValue = parsed;
          }
        }
        if (newValue !== null) {
          onChange(clamp(newValue));
        }
      }
      setInputValue("");
      setIsEditing(false);
    };
    const handleTextFocus = () => {
      setIsEditing(true);
      setInputValue(value.toString());
    };
    const handleKeyDown = (e) => {
      if (e.key === "Enter") {
        e.target.blur();
      } else if (e.key === "Escape") {
        setInputValue("");
        setIsEditing(false);
        e.target.blur();
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        handleIncrement(e.shiftKey ? actualCoarseStep : step);
      } else if (e.key === "ArrowDown") {
        e.preventDefault();
        handleIncrement(e.shiftKey ? -actualCoarseStep : -step);
      }
    };
    return /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 2, className: clsx("slider-container", className), children: [
      /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { justify: "between", align: "center", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("label", { className: "form-label mb-0 fw-semibold", children: /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { align: "center", gap: 1, children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: label }),
          info && /* @__PURE__ */ jsxRuntimeExports.jsx("span", { title: info, style: { cursor: "help", display: "flex", alignItems: "center" }, children: /* @__PURE__ */ jsxRuntimeExports.jsx(Info, { size: 14, className: "text-muted" }) })
        ] }) }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { align: "center", gap: 1, children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            Button,
            {
              use: "secondary",
              size: "sm",
              appearance: "outline",
              onClick: () => handleIncrement(-actualCoarseStep),
              disabled: disabled || value <= min,
              title: `Decrease by ${actualCoarseStep} ${unit}`,
              className: "px-1 py-0",
              children: /* @__PURE__ */ jsxRuntimeExports.jsx(ChevronsDown, { size: 14 })
            }
          ),
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            Button,
            {
              use: "secondary",
              size: "sm",
              appearance: "outline",
              onClick: () => handleIncrement(-step),
              disabled: disabled || value <= min,
              title: `Decrease by ${step} ${unit}`,
              className: "px-1 py-0",
              children: /* @__PURE__ */ jsxRuntimeExports.jsx(ChevronDown, { size: 14 })
            }
          ),
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "input",
            {
              type: "text",
              className: "form-control form-control-sm text-center",
              style: { width: "110px", fontFamily: "monospace", fontSize: "0.875rem" },
              value: isEditing ? inputValue : displayValue,
              onChange: (e) => handleTextChange(e.target.value),
              onFocus: handleTextFocus,
              onBlur: handleTextBlur,
              onKeyDown: handleKeyDown,
              disabled,
              placeholder: displayValue
            }
          ),
          /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "small text-muted", children: unit }),
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            Button,
            {
              use: "secondary",
              size: "sm",
              appearance: "outline",
              onClick: () => handleIncrement(step),
              disabled: disabled || value >= max,
              title: `Increase by ${step} ${unit}`,
              className: "px-1 py-0",
              children: /* @__PURE__ */ jsxRuntimeExports.jsx(ChevronUp, { size: 14 })
            }
          ),
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            Button,
            {
              use: "secondary",
              size: "sm",
              appearance: "outline",
              onClick: () => handleIncrement(actualCoarseStep),
              disabled: disabled || value >= max,
              title: `Increase by ${actualCoarseStep} ${unit}`,
              className: "px-1 py-0",
              children: /* @__PURE__ */ jsxRuntimeExports.jsx(ChevronsUp, { size: 14 })
            }
          )
        ] })
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsx(
        "input",
        {
          ...rest,
          ref,
          type: "range",
          className: "form-range",
          min,
          max,
          step,
          value,
          disabled,
          onChange: (e) => onChange(parseFloat(e.target.value))
        }
      ),
      showMinMax && /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { justify: "between", className: "small text-muted", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { children: [
          formatValue ? formatValue(min) : min.toLocaleString(),
          " ",
          unit
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { children: [
          formatValue ? formatValue(max) : max.toLocaleString(),
          " ",
          unit
        ] })
      ] })
    ] });
  }
);
Slider.displayName = "Slider";
function NumericSelector({
  label,
  value,
  min,
  max,
  step = 1,
  info,
  onChange,
  disabled = false,
  units
}) {
  const [isEditing, setIsEditing] = reactExports.useState(false);
  const [inputValue, setInputValue] = reactExports.useState("");
  const [currentUnitIndex, setCurrentUnitIndex] = reactExports.useState(0);
  const currentUnit = units == null ? void 0 : units[currentUnitIndex];
  const displayValue = currentUnit ? value / currentUnit.multiplier : value;
  const unitMultiplier = (currentUnit == null ? void 0 : currentUnit.multiplier) ?? 1;
  const unitName = (currentUnit == null ? void 0 : currentUnit.name) ?? "";
  const decimals = (currentUnit == null ? void 0 : currentUnit.decimals) ?? 0;
  const placeValues = (currentUnit == null ? void 0 : currentUnit.placeValues) ?? [];
  const formatValue = (val) => {
    return val.toFixed(decimals);
  };
  const clamp = (val) => Math.max(min, Math.min(max, val));
  const adjustByPlace = (placeValue, direction) => {
    const delta = placeValue * direction;
    onChange(clamp(value + delta));
  };
  const handleSliderChange = (e) => {
    onChange(parseFloat(e.target.value));
  };
  const handleTextFocus = () => {
    setIsEditing(true);
    setInputValue(formatValue(displayValue));
  };
  const handleTextBlur = () => {
    if (inputValue) {
      const parsed = parseFloat(inputValue);
      if (!isNaN(parsed)) {
        onChange(clamp(parsed * unitMultiplier));
      }
    }
    setInputValue("");
    setIsEditing(false);
  };
  const handleKeyDown = (e) => {
    if (e.key === "Enter") {
      e.target.blur();
    } else if (e.key === "Escape") {
      setInputValue("");
      setIsEditing(false);
      e.target.blur();
    }
  };
  return /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 2, children: [
    /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { justify: "between", align: "center", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("label", { className: "form-label mb-0 fw-semibold", children: /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { align: "center", gap: 1, children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: label }),
        info && /* @__PURE__ */ jsxRuntimeExports.jsx("span", { title: info, style: { cursor: "help", display: "flex", alignItems: "center" }, children: /* @__PURE__ */ jsxRuntimeExports.jsx(Info, { size: 14, className: "text-muted" }) })
      ] }) }),
      units && units.length > 1 && /* @__PURE__ */ jsxRuntimeExports.jsx(
        "select",
        {
          className: "form-select form-select-sm",
          style: { width: "auto" },
          value: currentUnitIndex,
          onChange: (e) => setCurrentUnitIndex(parseInt(e.target.value)),
          disabled,
          children: units.map((unit, idx) => /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: idx, children: unit.name }, idx))
        }
      )
    ] }),
    /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { gap: 3, align: "center", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("div", { style: { flex: 1 }, children: /* @__PURE__ */ jsxRuntimeExports.jsx(
        "input",
        {
          type: "range",
          className: "form-range",
          min,
          max,
          step,
          value,
          disabled,
          onChange: handleSliderChange
        }
      ) }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { gap: 1, align: "center", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx(
          "input",
          {
            type: "text",
            className: "form-control form-control-sm text-end",
            style: { width: "85px", fontFamily: "monospace", fontSize: "0.875rem" },
            value: isEditing ? inputValue : formatValue(displayValue),
            onChange: (e) => setInputValue(e.target.value),
            onFocus: handleTextFocus,
            onBlur: handleTextBlur,
            onKeyDown: handleKeyDown,
            disabled
          }
        ),
        /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "small text-muted", style: { width: "40px" }, children: unitName }),
        placeValues.map((place, idx) => /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { align: "center", gap: 0, children: [
          /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 0, style: { width: "fit-content", minWidth: "28px" }, children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "text-center text-muted", style: { fontSize: "8px", lineHeight: "10px", marginBottom: "1px" }, children: place.label }),
            /* @__PURE__ */ jsxRuntimeExports.jsx(
              "button",
              {
                type: "button",
                className: "btn btn-sm btn-outline-secondary p-0",
                style: { height: "16px", lineHeight: "1", fontSize: "10px", borderRadius: "2px 2px 0 0" },
                onClick: () => adjustByPlace(place.value, 1),
                disabled: disabled || value >= max,
                title: `+${place.label} ${unitName}`,
                children: /* @__PURE__ */ jsxRuntimeExports.jsx(ChevronUp, { size: 12, style: { marginTop: "-2px" } })
              }
            ),
            /* @__PURE__ */ jsxRuntimeExports.jsx(
              "button",
              {
                type: "button",
                className: "btn btn-sm btn-outline-secondary p-0",
                style: { height: "16px", lineHeight: "1", fontSize: "10px", borderRadius: "0 0 2px 2px", borderTop: "none" },
                onClick: () => adjustByPlace(place.value, -1),
                disabled: disabled || value <= min,
                title: `-${place.label} ${unitName}`,
                children: /* @__PURE__ */ jsxRuntimeExports.jsx(ChevronDown, { size: 12, style: { marginTop: "-2px" } })
              }
            )
          ] }),
          place.label.includes(".") && idx < placeValues.length - 1 && !placeValues[idx + 1].label.includes(".") && /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "text-muted", style: { fontSize: "10px", marginLeft: "2px", marginRight: "2px" }, children: "." })
        ] }, idx))
      ] })
    ] }),
    /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { justify: "between", className: "small text-muted", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { children: [
        formatValue(min / unitMultiplier),
        " ",
        unitName
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { children: [
        formatValue(max / unitMultiplier),
        " ",
        unitName
      ] })
    ] })
  ] });
}
const frequencyUnits = [
  {
    name: "MHz",
    multiplier: 1e6,
    decimals: 3,
    placeValues: [
      { label: "100", value: 1e8 },
      { label: "10", value: 1e7 },
      { label: "1", value: 1e6 },
      { label: "0.1", value: 1e5 },
      { label: "0.01", value: 1e4 },
      { label: "0.001", value: 1e3 }
    ]
  },
  {
    name: "kHz",
    multiplier: 1e3,
    decimals: 1,
    placeValues: [
      { label: "10000", value: 1e10 },
      { label: "1000", value: 1e9 },
      { label: "100", value: 1e8 },
      { label: "10", value: 1e7 },
      { label: "1", value: 1e6 },
      { label: "0.1", value: 1e5 }
    ]
  }
];
function FrequencySelector(props) {
  return /* @__PURE__ */ jsxRuntimeExports.jsx(NumericSelector, { ...props, units: frequencyUnits });
}
const Spinner = ({
  size = "md",
  variant = "primary",
  label = "Loading…",
  className
}) => {
  const sizeClass = size === "sm" ? "spinner-border-sm" : null;
  const variantClass = variant === "inherit" ? null : `text-${variant}`;
  return /* @__PURE__ */ jsxRuntimeExports.jsx(
    "span",
    {
      role: "status",
      "aria-live": "polite",
      "aria-busy": "true",
      className: clsx("spinner-border", sizeClass, variantClass, className),
      children: /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "visually-hidden", children: label })
    }
  );
};
const STORAGE_KEY$2 = "wavecap_frequency_bookmarks";
function useBookmarks() {
  const [bookmarks, setBookmarks] = reactExports.useState(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY$2);
      return stored ? JSON.parse(stored) : [];
    } catch (error) {
      console.error("Failed to load bookmarks:", error);
      return [];
    }
  });
  reactExports.useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY$2, JSON.stringify(bookmarks));
    } catch (error) {
      console.error("Failed to save bookmarks:", error);
    }
  }, [bookmarks]);
  const addBookmark = (bookmark) => {
    const newBookmark = {
      ...bookmark,
      id: `bookmark_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      createdAt: Date.now()
    };
    setBookmarks((prev) => [...prev, newBookmark]);
    return newBookmark;
  };
  const updateBookmark = (id2, updates) => {
    setBookmarks(
      (prev) => prev.map((b) => b.id === id2 ? { ...b, ...updates } : b)
    );
  };
  const deleteBookmark = (id2) => {
    setBookmarks((prev) => prev.filter((b) => b.id !== id2));
  };
  const getBookmarkByFrequency = (frequency, tolerance = 1e3) => {
    return bookmarks.find((b) => Math.abs(b.frequency - frequency) < tolerance);
  };
  return {
    bookmarks,
    addBookmark,
    updateBookmark,
    deleteBookmark,
    getBookmarkByFrequency
  };
}
const HISTORY_KEY = "wavecapsdr_frequency_history";
const MAX_HISTORY_ENTRIES = 100;
function useFrequencyHistory() {
  const [history, setHistory] = reactExports.useState([]);
  reactExports.useEffect(() => {
    try {
      const stored = localStorage.getItem(HISTORY_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        setHistory(parsed);
      }
    } catch (error) {
      console.error("Failed to load frequency history:", error);
    }
  }, []);
  const saveHistory = reactExports.useCallback((newHistory) => {
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(newHistory));
      setHistory(newHistory);
    } catch (error) {
      console.error("Failed to save frequency history:", error);
    }
  }, []);
  const addToHistory = reactExports.useCallback((entry) => {
    const newEntry = {
      ...entry,
      timestamp: Date.now()
    };
    const fiveMinutesAgo = Date.now() - 5 * 60 * 1e3;
    const filteredHistory = history.filter(
      (h) => h.frequencyHz !== entry.frequencyHz || h.timestamp < fiveMinutesAgo
    );
    const newHistory = [newEntry, ...filteredHistory].slice(0, MAX_HISTORY_ENTRIES);
    saveHistory(newHistory);
  }, [history, saveHistory]);
  const clearHistory = reactExports.useCallback(() => {
    try {
      localStorage.removeItem(HISTORY_KEY);
      setHistory([]);
    } catch (error) {
      console.error("Failed to clear frequency history:", error);
    }
  }, []);
  const getRecentHistory = reactExports.useCallback((limit = 20) => {
    return history.slice(0, limit);
  }, [history]);
  return {
    history,
    addToHistory,
    clearHistory,
    getRecentHistory
  };
}
const BookmarkManager = ({ currentFrequency, onTuneToFrequency, currentCapture, currentChannels, onLoadMemoryBank }) => {
  const { bookmarks, addBookmark, updateBookmark, deleteBookmark, getBookmarkByFrequency } = useBookmarks();
  const { getRecentHistory, addToHistory } = useFrequencyHistory();
  const { memoryBanks, saveToMemoryBank, deleteMemoryBank } = useMemoryBanks();
  const [showDropdown, setShowDropdown] = reactExports.useState(false);
  const [activeTab, setActiveTab] = reactExports.useState("bookmarks");
  const [showAddModal, setShowAddModal] = reactExports.useState(false);
  const [showSaveMemoryModal, setShowSaveMemoryModal] = reactExports.useState(false);
  const [memoryBankName, setMemoryBankName] = reactExports.useState("");
  const [editingBookmark, setEditingBookmark] = reactExports.useState(null);
  const [bookmarkName, setBookmarkName] = reactExports.useState("");
  const [bookmarkNotes, setBookmarkNotes] = reactExports.useState("");
  const recentHistory = getRecentHistory(20);
  const currentBookmark = getBookmarkByFrequency(currentFrequency);
  const isBookmarked = currentBookmark !== void 0;
  const handleAddBookmark = () => {
    if (!bookmarkName.trim())
      return;
    addBookmark({
      name: bookmarkName.trim(),
      frequency: currentFrequency,
      notes: bookmarkNotes.trim() || void 0
    });
    setBookmarkName("");
    setBookmarkNotes("");
    setShowAddModal(false);
  };
  const handleEditBookmark = (bookmark) => {
    setEditingBookmark(bookmark);
    setBookmarkName(bookmark.name);
    setBookmarkNotes(bookmark.notes || "");
    setShowAddModal(true);
  };
  const handleUpdateBookmark = () => {
    if (!editingBookmark || !bookmarkName.trim())
      return;
    updateBookmark(editingBookmark.id, {
      name: bookmarkName.trim(),
      notes: bookmarkNotes.trim() || void 0
    });
    setEditingBookmark(null);
    setBookmarkName("");
    setBookmarkNotes("");
    setShowAddModal(false);
  };
  const handleDeleteBookmark = (id2) => {
    deleteBookmark(id2);
  };
  const handleTuneToBookmark = (bookmark) => {
    onTuneToFrequency(bookmark.frequency);
    addToHistory({ frequencyHz: bookmark.frequency });
    setShowDropdown(false);
  };
  const handleTuneToRecent = (frequencyHz) => {
    onTuneToFrequency(frequencyHz);
    setShowDropdown(false);
  };
  const handleSaveMemoryBank = () => {
    if (!memoryBankName.trim() || !currentCapture || !currentChannels)
      return;
    saveToMemoryBank(memoryBankName.trim(), currentCapture, currentChannels);
    setMemoryBankName("");
    setShowSaveMemoryModal(false);
  };
  const formatFrequency = (hz) => {
    if (hz >= 1e9) {
      return `${(hz / 1e9).toFixed(3)} GHz`;
    } else if (hz >= 1e6) {
      return `${(hz / 1e6).toFixed(3)} MHz`;
    } else if (hz >= 1e3) {
      return `${(hz / 1e3).toFixed(3)} kHz`;
    }
    return `${hz} Hz`;
  };
  return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { style: { position: "relative" }, children: [
    /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { gap: 1, children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx(
        Button,
        {
          use: isBookmarked ? "warning" : "secondary",
          size: "sm",
          onClick: () => {
            if (isBookmarked) {
              handleDeleteBookmark(currentBookmark.id);
            } else {
              setShowAddModal(true);
            }
          },
          title: isBookmarked ? "Remove bookmark" : "Add bookmark",
          children: /* @__PURE__ */ jsxRuntimeExports.jsx(Star, { size: 16, fill: isBookmarked ? "currentColor" : "none" })
        }
      ),
      /* @__PURE__ */ jsxRuntimeExports.jsx(
        Button,
        {
          use: "secondary",
          size: "sm",
          onClick: () => setShowDropdown(!showDropdown),
          title: "View bookmarks, history, and memory banks",
          children: /* @__PURE__ */ jsxRuntimeExports.jsx(Clock, { size: 16 })
        }
      ),
      currentCapture && currentChannels && currentChannels.length > 0 && /* @__PURE__ */ jsxRuntimeExports.jsx(
        Button,
        {
          use: "primary",
          size: "sm",
          onClick: () => setShowSaveMemoryModal(true),
          title: "Save current configuration to memory bank",
          children: /* @__PURE__ */ jsxRuntimeExports.jsx(Save, { size: 16 })
        }
      )
    ] }),
    showDropdown && /* @__PURE__ */ jsxRuntimeExports.jsxs(
      "div",
      {
        style: {
          position: "absolute",
          top: "100%",
          right: 0,
          marginTop: "4px",
          backgroundColor: "white",
          border: "1px solid #dee2e6",
          borderRadius: "4px",
          boxShadow: "0 2px 8px rgba(0,0,0,0.15)",
          minWidth: "350px",
          maxHeight: "450px",
          display: "flex",
          flexDirection: "column",
          zIndex: 1e3
        },
        children: [
          /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { style: {
            display: "flex",
            borderBottom: "2px solid #dee2e6",
            backgroundColor: "#f8f9fa"
          }, children: [
            /* @__PURE__ */ jsxRuntimeExports.jsxs(
              "button",
              {
                className: `btn btn-sm ${activeTab === "bookmarks" ? "btn-primary" : "btn-light"}`,
                style: {
                  flex: 1,
                  borderRadius: 0,
                  borderTopLeftRadius: "4px",
                  fontSize: "12px"
                },
                onClick: () => setActiveTab("bookmarks"),
                children: [
                  "Bookmarks (",
                  bookmarks.length,
                  ")"
                ]
              }
            ),
            /* @__PURE__ */ jsxRuntimeExports.jsxs(
              "button",
              {
                className: `btn btn-sm ${activeTab === "recent" ? "btn-primary" : "btn-light"}`,
                style: {
                  flex: 1,
                  borderRadius: 0,
                  fontSize: "12px"
                },
                onClick: () => setActiveTab("recent"),
                children: [
                  "Recent (",
                  recentHistory.length,
                  ")"
                ]
              }
            ),
            /* @__PURE__ */ jsxRuntimeExports.jsxs(
              "button",
              {
                className: `btn btn-sm ${activeTab === "memory" ? "btn-primary" : "btn-light"}`,
                style: {
                  flex: 1,
                  borderRadius: 0,
                  borderTopRightRadius: "4px",
                  fontSize: "12px"
                },
                onClick: () => setActiveTab("memory"),
                children: [
                  "Memory (",
                  memoryBanks.length,
                  ")"
                ]
              }
            )
          ] }),
          /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { style: { overflowY: "auto", maxHeight: "380px" }, children: [
            activeTab === "bookmarks" && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { children: bookmarks.length === 0 ? /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { style: { padding: "20px", textAlign: "center", color: "#6c757d" }, children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx(Star, { size: 32, style: { opacity: 0.3, marginBottom: "8px" } }),
              /* @__PURE__ */ jsxRuntimeExports.jsx("div", { style: { fontSize: "12px" }, children: "No bookmarks yet" }),
              /* @__PURE__ */ jsxRuntimeExports.jsx("div", { style: { fontSize: "11px", marginTop: "4px" }, children: "Click the star icon to add a bookmark" })
            ] }) : bookmarks.map((bookmark) => /* @__PURE__ */ jsxRuntimeExports.jsx(
              "div",
              {
                style: {
                  padding: "8px 12px",
                  borderBottom: "1px solid #f0f0f0",
                  cursor: "pointer"
                },
                onMouseEnter: (e) => {
                  e.currentTarget.style.backgroundColor = "#f8f9fa";
                },
                onMouseLeave: (e) => {
                  e.currentTarget.style.backgroundColor = "white";
                },
                children: /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { justify: "between", align: "center", children: [
                  /* @__PURE__ */ jsxRuntimeExports.jsxs(
                    "div",
                    {
                      style: { flex: 1 },
                      onClick: () => handleTuneToBookmark(bookmark),
                      children: [
                        /* @__PURE__ */ jsxRuntimeExports.jsx("div", { style: { fontWeight: 500 }, children: bookmark.name }),
                        /* @__PURE__ */ jsxRuntimeExports.jsx("div", { style: { fontSize: "12px", color: "#6c757d" }, children: formatFrequency(bookmark.frequency) }),
                        bookmark.notes && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { style: { fontSize: "12px", color: "#6c757d", marginTop: "2px" }, children: bookmark.notes })
                      ]
                    }
                  ),
                  /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { gap: 1, children: [
                    /* @__PURE__ */ jsxRuntimeExports.jsx(
                      "button",
                      {
                        className: "btn btn-sm btn-icon-sm",
                        style: { padding: "2px 6px" },
                        onClick: (e) => {
                          e.stopPropagation();
                          handleEditBookmark(bookmark);
                          setShowDropdown(false);
                        },
                        title: "Edit bookmark",
                        children: /* @__PURE__ */ jsxRuntimeExports.jsx(Pen, { size: 14 })
                      }
                    ),
                    /* @__PURE__ */ jsxRuntimeExports.jsx(
                      "button",
                      {
                        className: "btn btn-sm btn-danger btn-icon-sm",
                        style: { padding: "2px 6px" },
                        onClick: (e) => {
                          e.stopPropagation();
                          handleDeleteBookmark(bookmark.id);
                        },
                        title: "Delete bookmark",
                        children: /* @__PURE__ */ jsxRuntimeExports.jsx(Trash2, { size: 14 })
                      }
                    )
                  ] })
                ] })
              },
              bookmark.id
            )) }),
            activeTab === "recent" && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { children: recentHistory.length === 0 ? /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { style: { padding: "20px", textAlign: "center", color: "#6c757d" }, children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx(Clock, { size: 32, style: { opacity: 0.3, marginBottom: "8px" } }),
              /* @__PURE__ */ jsxRuntimeExports.jsx("div", { style: { fontSize: "12px" }, children: "No recent history" })
            ] }) : recentHistory.map((entry) => /* @__PURE__ */ jsxRuntimeExports.jsx(
              "div",
              {
                style: {
                  padding: "8px 12px",
                  borderBottom: "1px solid #f0f0f0",
                  cursor: "pointer"
                },
                onMouseEnter: (e) => {
                  e.currentTarget.style.backgroundColor = "#f8f9fa";
                },
                onMouseLeave: (e) => {
                  e.currentTarget.style.backgroundColor = "white";
                },
                onClick: () => handleTuneToRecent(entry.frequencyHz),
                children: /* @__PURE__ */ jsxRuntimeExports.jsx(Flex, { justify: "between", align: "center", children: /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { style: { flex: 1 }, children: [
                  /* @__PURE__ */ jsxRuntimeExports.jsx("div", { style: { fontWeight: 500, fontSize: "14px" }, children: formatFrequency(entry.frequencyHz) }),
                  /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { style: { fontSize: "11px", color: "#6c757d" }, children: [
                    new Date(entry.timestamp).toLocaleString(),
                    entry.captureName && ` • ${entry.captureName}`,
                    entry.mode && ` • ${entry.mode.toUpperCase()}`
                  ] })
                ] }) })
              },
              `${entry.frequencyHz}-${entry.timestamp}`
            )) }),
            activeTab === "memory" && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { children: memoryBanks.length === 0 ? /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { style: { padding: "20px", textAlign: "center", color: "#6c757d" }, children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx(Save, { size: 32, style: { opacity: 0.3, marginBottom: "8px" } }),
              /* @__PURE__ */ jsxRuntimeExports.jsx("div", { style: { fontSize: "12px" }, children: "No saved memory banks" }),
              /* @__PURE__ */ jsxRuntimeExports.jsx("div", { style: { fontSize: "11px", marginTop: "4px" }, children: "Click the save icon to save current configuration" })
            ] }) : memoryBanks.map((bank) => /* @__PURE__ */ jsxRuntimeExports.jsx(
              "div",
              {
                style: {
                  padding: "8px 12px",
                  borderBottom: "1px solid #f0f0f0"
                },
                children: /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { justify: "between", align: "center", children: [
                  /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { style: { flex: 1 }, children: [
                    /* @__PURE__ */ jsxRuntimeExports.jsx("div", { style: { fontWeight: 500 }, children: bank.name }),
                    /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { style: { fontSize: "11px", color: "#6c757d" }, children: [
                      formatFrequency(bank.captureConfig.centerHz),
                      " •",
                      (bank.captureConfig.sampleRate / 1e6).toFixed(1),
                      " MS/s •",
                      bank.channels.length,
                      " channel",
                      bank.channels.length !== 1 ? "s" : ""
                    ] }),
                    /* @__PURE__ */ jsxRuntimeExports.jsx("div", { style: { fontSize: "10px", color: "#6c757d", marginTop: "2px" }, children: new Date(bank.timestamp).toLocaleString() })
                  ] }),
                  /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { gap: 1, children: [
                    onLoadMemoryBank && /* @__PURE__ */ jsxRuntimeExports.jsx(
                      "button",
                      {
                        className: "btn btn-sm btn-primary",
                        style: { padding: "2px 8px", fontSize: "11px" },
                        onClick: () => {
                          onLoadMemoryBank(bank.id);
                          setShowDropdown(false);
                        },
                        title: "Load this memory bank",
                        children: "Load"
                      }
                    ),
                    /* @__PURE__ */ jsxRuntimeExports.jsx(
                      "button",
                      {
                        className: "btn btn-sm btn-danger btn-icon-sm",
                        style: { padding: "2px 6px" },
                        onClick: (e) => {
                          e.stopPropagation();
                          if (confirm(`Delete memory bank "${bank.name}"?`)) {
                            deleteMemoryBank(bank.id);
                          }
                        },
                        title: "Delete memory bank",
                        children: /* @__PURE__ */ jsxRuntimeExports.jsx(Trash2, { size: 14 })
                      }
                    )
                  ] })
                ] })
              },
              bank.id
            )) })
          ] })
        ]
      }
    ),
    showAddModal && /* @__PURE__ */ jsxRuntimeExports.jsx(
      "div",
      {
        style: {
          position: "fixed",
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: "rgba(0,0,0,0.5)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          zIndex: 2e3
        },
        onClick: () => {
          setShowAddModal(false);
          setEditingBookmark(null);
          setBookmarkName("");
          setBookmarkNotes("");
        },
        children: /* @__PURE__ */ jsxRuntimeExports.jsxs(
          "div",
          {
            style: {
              backgroundColor: "white",
              borderRadius: "8px",
              padding: "20px",
              maxWidth: "400px",
              width: "90%"
            },
            onClick: (e) => e.stopPropagation(),
            children: [
              /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { justify: "between", align: "center", className: "mb-3", children: [
                /* @__PURE__ */ jsxRuntimeExports.jsx("h5", { className: "mb-0", children: editingBookmark ? "Edit Bookmark" : "Add Bookmark" }),
                /* @__PURE__ */ jsxRuntimeExports.jsx(
                  "button",
                  {
                    className: "btn btn-sm btn-icon-sm",
                    onClick: () => {
                      setShowAddModal(false);
                      setEditingBookmark(null);
                      setBookmarkName("");
                      setBookmarkNotes("");
                    },
                    children: /* @__PURE__ */ jsxRuntimeExports.jsx(X, { size: 20 })
                  }
                )
              ] }),
              /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "mb-3", children: [
                /* @__PURE__ */ jsxRuntimeExports.jsx("label", { className: "form-label small", children: "Name" }),
                /* @__PURE__ */ jsxRuntimeExports.jsx(
                  "input",
                  {
                    type: "text",
                    className: "form-control form-control-sm",
                    value: bookmarkName,
                    onChange: (e) => setBookmarkName(e.target.value),
                    placeholder: "e.g., KEXP 90.3 FM",
                    autoFocus: true
                  }
                )
              ] }),
              /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "mb-3", children: [
                /* @__PURE__ */ jsxRuntimeExports.jsx("label", { className: "form-label small", children: "Frequency" }),
                /* @__PURE__ */ jsxRuntimeExports.jsx(
                  "input",
                  {
                    type: "text",
                    className: "form-control form-control-sm",
                    value: formatFrequency((editingBookmark == null ? void 0 : editingBookmark.frequency) || currentFrequency),
                    disabled: true
                  }
                )
              ] }),
              /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "mb-3", children: [
                /* @__PURE__ */ jsxRuntimeExports.jsx("label", { className: "form-label small", children: "Notes (optional)" }),
                /* @__PURE__ */ jsxRuntimeExports.jsx(
                  "textarea",
                  {
                    className: "form-control form-control-sm",
                    value: bookmarkNotes,
                    onChange: (e) => setBookmarkNotes(e.target.value),
                    placeholder: "e.g., Local indie radio station",
                    rows: 2
                  }
                )
              ] }),
              /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { gap: 2, justify: "end", children: [
                /* @__PURE__ */ jsxRuntimeExports.jsx(
                  Button,
                  {
                    use: "secondary",
                    size: "sm",
                    onClick: () => {
                      setShowAddModal(false);
                      setEditingBookmark(null);
                      setBookmarkName("");
                      setBookmarkNotes("");
                    },
                    children: "Cancel"
                  }
                ),
                /* @__PURE__ */ jsxRuntimeExports.jsx(
                  Button,
                  {
                    use: "primary",
                    size: "sm",
                    onClick: editingBookmark ? handleUpdateBookmark : handleAddBookmark,
                    disabled: !bookmarkName.trim(),
                    children: editingBookmark ? "Update" : "Add"
                  }
                )
              ] })
            ]
          }
        )
      }
    ),
    showSaveMemoryModal && /* @__PURE__ */ jsxRuntimeExports.jsx(
      "div",
      {
        style: {
          position: "fixed",
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: "rgba(0,0,0,0.5)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          zIndex: 2e3
        },
        onClick: () => {
          setShowSaveMemoryModal(false);
          setMemoryBankName("");
        },
        children: /* @__PURE__ */ jsxRuntimeExports.jsxs(
          "div",
          {
            style: {
              backgroundColor: "white",
              borderRadius: "8px",
              padding: "20px",
              maxWidth: "400px",
              width: "90%"
            },
            onClick: (e) => e.stopPropagation(),
            children: [
              /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { justify: "between", align: "center", className: "mb-3", children: [
                /* @__PURE__ */ jsxRuntimeExports.jsx("h5", { className: "mb-0", children: "Save to Memory Bank" }),
                /* @__PURE__ */ jsxRuntimeExports.jsx(
                  "button",
                  {
                    className: "btn btn-sm btn-icon-sm",
                    onClick: () => {
                      setShowSaveMemoryModal(false);
                      setMemoryBankName("");
                    },
                    children: /* @__PURE__ */ jsxRuntimeExports.jsx(X, { size: 20 })
                  }
                )
              ] }),
              /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "mb-3", children: [
                /* @__PURE__ */ jsxRuntimeExports.jsx("label", { className: "form-label small", children: "Name" }),
                /* @__PURE__ */ jsxRuntimeExports.jsx(
                  "input",
                  {
                    type: "text",
                    className: "form-control form-control-sm",
                    value: memoryBankName,
                    onChange: (e) => setMemoryBankName(e.target.value),
                    placeholder: "e.g., Local Public Safety, Ham Bands",
                    autoFocus: true
                  }
                )
              ] }),
              currentCapture && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "mb-3", children: [
                /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "small text-muted", children: "This will save:" }),
                /* @__PURE__ */ jsxRuntimeExports.jsxs("ul", { className: "small mb-0", style: { paddingLeft: "20px" }, children: [
                  /* @__PURE__ */ jsxRuntimeExports.jsxs("li", { children: [
                    "Center: ",
                    formatFrequency(currentCapture.centerHz)
                  ] }),
                  /* @__PURE__ */ jsxRuntimeExports.jsxs("li", { children: [
                    "Sample Rate: ",
                    (currentCapture.sampleRate / 1e6).toFixed(1),
                    " MS/s"
                  ] }),
                  /* @__PURE__ */ jsxRuntimeExports.jsxs("li", { children: [
                    "Channels: ",
                    (currentChannels == null ? void 0 : currentChannels.length) || 0
                  ] }),
                  currentCapture.gain !== null && /* @__PURE__ */ jsxRuntimeExports.jsxs("li", { children: [
                    "Gain: ",
                    currentCapture.gain,
                    " dB"
                  ] })
                ] })
              ] }),
              /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { gap: 2, justify: "end", children: [
                /* @__PURE__ */ jsxRuntimeExports.jsx(
                  Button,
                  {
                    use: "secondary",
                    size: "sm",
                    onClick: () => {
                      setShowSaveMemoryModal(false);
                      setMemoryBankName("");
                    },
                    children: "Cancel"
                  }
                ),
                /* @__PURE__ */ jsxRuntimeExports.jsx(
                  Button,
                  {
                    use: "primary",
                    size: "sm",
                    onClick: handleSaveMemoryBank,
                    disabled: !memoryBankName.trim(),
                    children: "Save"
                  }
                )
              ] })
            ]
          }
        )
      }
    )
  ] });
};
const API_BASE$2 = "/api/v1";
async function fetchScanners() {
  const response = await fetch(`${API_BASE$2}/scanners`);
  if (!response.ok) {
    throw new Error(`Failed to fetch scanners: ${response.statusText}`);
  }
  return response.json();
}
async function createScanner(request) {
  const response = await fetch(`${API_BASE$2}/scanners`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request)
  });
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || `Failed to create scanner: ${response.statusText}`);
  }
  return response.json();
}
async function deleteScanner(scannerId) {
  const response = await fetch(`${API_BASE$2}/scanners/${scannerId}`, {
    method: "DELETE"
  });
  if (!response.ok) {
    throw new Error(`Failed to delete scanner: ${response.statusText}`);
  }
}
async function startScanner(scannerId) {
  const response = await fetch(`${API_BASE$2}/scanners/${scannerId}/start`, {
    method: "POST"
  });
  if (!response.ok) {
    throw new Error(`Failed to start scanner: ${response.statusText}`);
  }
  return response.json();
}
async function stopScanner(scannerId) {
  const response = await fetch(`${API_BASE$2}/scanners/${scannerId}/stop`, {
    method: "POST"
  });
  if (!response.ok) {
    throw new Error(`Failed to stop scanner: ${response.statusText}`);
  }
  return response.json();
}
async function pauseScanner(scannerId) {
  const response = await fetch(`${API_BASE$2}/scanners/${scannerId}/pause`, {
    method: "POST"
  });
  if (!response.ok) {
    throw new Error(`Failed to pause scanner: ${response.statusText}`);
  }
  return response.json();
}
async function resumeScanner(scannerId) {
  const response = await fetch(`${API_BASE$2}/scanners/${scannerId}/resume`, {
    method: "POST"
  });
  if (!response.ok) {
    throw new Error(`Failed to resume scanner: ${response.statusText}`);
  }
  return response.json();
}
async function lockScanner(scannerId) {
  const response = await fetch(`${API_BASE$2}/scanners/${scannerId}/lock`, {
    method: "POST"
  });
  if (!response.ok) {
    throw new Error(`Failed to lock scanner: ${response.statusText}`);
  }
  return response.json();
}
async function unlockScanner(scannerId) {
  const response = await fetch(`${API_BASE$2}/scanners/${scannerId}/unlock`, {
    method: "POST"
  });
  if (!response.ok) {
    throw new Error(`Failed to unlock scanner: ${response.statusText}`);
  }
  return response.json();
}
async function lockoutFrequency(scannerId) {
  const response = await fetch(`${API_BASE$2}/scanners/${scannerId}/lockout`, {
    method: "POST"
  });
  if (!response.ok) {
    throw new Error(`Failed to lockout frequency: ${response.statusText}`);
  }
  return response.json();
}
async function clearLockout(scannerId, frequency) {
  const response = await fetch(`${API_BASE$2}/scanners/${scannerId}/lockout/${frequency}`, {
    method: "DELETE"
  });
  if (!response.ok) {
    throw new Error(`Failed to clear lockout: ${response.statusText}`);
  }
  return response.json();
}
async function clearAllLockouts(scannerId) {
  const response = await fetch(`${API_BASE$2}/scanners/${scannerId}/lockouts`, {
    method: "DELETE"
  });
  if (!response.ok) {
    throw new Error(`Failed to clear all lockouts: ${response.statusText}`);
  }
  return response.json();
}
function useScanners() {
  return useQuery({
    queryKey: ["scanners"],
    queryFn: fetchScanners,
    refetchInterval: 2e3
    // Poll every 2 seconds for scanner updates
  });
}
function useCreateScanner() {
  const queryClient2 = useQueryClient();
  return useMutation({
    mutationFn: createScanner,
    onSuccess: () => {
      queryClient2.invalidateQueries({ queryKey: ["scanners"] });
    }
  });
}
function useDeleteScanner() {
  const queryClient2 = useQueryClient();
  return useMutation({
    mutationFn: deleteScanner,
    onSuccess: () => {
      queryClient2.invalidateQueries({ queryKey: ["scanners"] });
    }
  });
}
function useStartScanner() {
  const queryClient2 = useQueryClient();
  return useMutation({
    mutationFn: startScanner,
    onSuccess: (_data, scannerId) => {
      queryClient2.invalidateQueries({ queryKey: ["scanners"] });
      queryClient2.invalidateQueries({ queryKey: ["scanners", scannerId] });
    }
  });
}
function useStopScanner() {
  const queryClient2 = useQueryClient();
  return useMutation({
    mutationFn: stopScanner,
    onSuccess: (_data, scannerId) => {
      queryClient2.invalidateQueries({ queryKey: ["scanners"] });
      queryClient2.invalidateQueries({ queryKey: ["scanners", scannerId] });
    }
  });
}
function usePauseScanner() {
  const queryClient2 = useQueryClient();
  return useMutation({
    mutationFn: pauseScanner,
    onSuccess: (_data, scannerId) => {
      queryClient2.invalidateQueries({ queryKey: ["scanners"] });
      queryClient2.invalidateQueries({ queryKey: ["scanners", scannerId] });
    }
  });
}
function useResumeScanner() {
  const queryClient2 = useQueryClient();
  return useMutation({
    mutationFn: resumeScanner,
    onSuccess: (_data, scannerId) => {
      queryClient2.invalidateQueries({ queryKey: ["scanners"] });
      queryClient2.invalidateQueries({ queryKey: ["scanners", scannerId] });
    }
  });
}
function useLockScanner() {
  const queryClient2 = useQueryClient();
  return useMutation({
    mutationFn: lockScanner,
    onSuccess: (_data, scannerId) => {
      queryClient2.invalidateQueries({ queryKey: ["scanners"] });
      queryClient2.invalidateQueries({ queryKey: ["scanners", scannerId] });
    }
  });
}
function useUnlockScanner() {
  const queryClient2 = useQueryClient();
  return useMutation({
    mutationFn: unlockScanner,
    onSuccess: (_data, scannerId) => {
      queryClient2.invalidateQueries({ queryKey: ["scanners"] });
      queryClient2.invalidateQueries({ queryKey: ["scanners", scannerId] });
    }
  });
}
function useLockoutFrequency() {
  const queryClient2 = useQueryClient();
  return useMutation({
    mutationFn: lockoutFrequency,
    onSuccess: (_data, scannerId) => {
      queryClient2.invalidateQueries({ queryKey: ["scanners"] });
      queryClient2.invalidateQueries({ queryKey: ["scanners", scannerId] });
    }
  });
}
function useClearLockout() {
  const queryClient2 = useQueryClient();
  return useMutation({
    mutationFn: ({ scannerId, frequency }) => clearLockout(scannerId, frequency),
    onSuccess: (_data, variables) => {
      queryClient2.invalidateQueries({ queryKey: ["scanners"] });
      queryClient2.invalidateQueries({ queryKey: ["scanners", variables.scannerId] });
    }
  });
}
function useClearAllLockouts() {
  const queryClient2 = useQueryClient();
  return useMutation({
    mutationFn: clearAllLockouts,
    onSuccess: (_data, scannerId) => {
      queryClient2.invalidateQueries({ queryKey: ["scanners"] });
      queryClient2.invalidateQueries({ queryKey: ["scanners", scannerId] });
    }
  });
}
function ScannerControl({ captureId }) {
  const { data: allScanners, isLoading } = useScanners();
  const createScanner2 = useCreateScanner();
  const deleteScanner2 = useDeleteScanner();
  const scanners = reactExports.useMemo(() => {
    return (allScanners == null ? void 0 : allScanners.filter((s) => s.captureId === captureId)) || [];
  }, [allScanners, captureId]);
  const [showCreateWizard, setShowCreateWizard] = reactExports.useState(false);
  const [selectedScannerId, setSelectedScannerId] = reactExports.useState(null);
  const scanner = scanners.find((s) => s.id === selectedScannerId);
  if (isLoading) {
    return /* @__PURE__ */ jsxRuntimeExports.jsx("div", { style: { padding: "12px", color: "#6c757d" }, children: "Loading scanners..." });
  }
  if (showCreateWizard) {
    return /* @__PURE__ */ jsxRuntimeExports.jsx(
      CreateScannerWizard,
      {
        captureId,
        onCancel: () => setShowCreateWizard(false),
        onCreate: (req) => {
          createScanner2.mutate(req, {
            onSuccess: (newScanner) => {
              setSelectedScannerId(newScanner.id);
              setShowCreateWizard(false);
            }
          });
        }
      }
    );
  }
  if (scanners.length === 0) {
    return /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 3, style: { padding: "16px" }, children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("div", { style: { color: "#6c757d", textAlign: "center" }, children: "No scanners configured." }),
      /* @__PURE__ */ jsxRuntimeExports.jsx(Button, { onClick: () => setShowCreateWizard(true), children: "Create Scanner" })
    ] });
  }
  if (!scanner) {
    return /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 3, style: { padding: "16px" }, children: [
      /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "row", gap: 2, align: "center", justify: "between", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("h3", { style: { margin: 0, fontSize: "14px", fontWeight: 600 }, children: "Scanners" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx(Button, { size: "sm", onClick: () => setShowCreateWizard(true), children: "New" })
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsx(Flex, { direction: "column", gap: 2, children: scanners.map((s) => /* @__PURE__ */ jsxRuntimeExports.jsx(
        "div",
        {
          onClick: () => setSelectedScannerId(s.id),
          style: {
            padding: "12px",
            border: "1px solid #dee2e6",
            borderRadius: "6px",
            cursor: "pointer",
            backgroundColor: "#f8f9fa"
          },
          children: /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "row", justify: "between", align: "center", children: [
            /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { children: [
              /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { style: { fontWeight: 600, fontSize: "13px" }, children: [
                "Scanner ",
                s.id
              ] }),
              /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { style: { fontSize: "11px", color: "#6c757d" }, children: [
                s.scanList.length,
                " freqs • ",
                s.mode,
                " • ",
                s.state
              ] })
            ] }),
            /* @__PURE__ */ jsxRuntimeExports.jsx(
              "div",
              {
                style: {
                  width: "10px",
                  height: "10px",
                  borderRadius: "50%",
                  backgroundColor: s.state === "scanning" ? "#28a745" : s.state === "paused" ? "#ffc107" : "#6c757d"
                }
              }
            )
          ] })
        },
        s.id
      )) })
    ] });
  }
  return /* @__PURE__ */ jsxRuntimeExports.jsx(
    ScannerDetail,
    {
      scanner,
      onBack: () => setSelectedScannerId(null),
      onDelete: () => {
        deleteScanner2.mutate(scanner.id, {
          onSuccess: () => setSelectedScannerId(null)
        });
      }
    }
  );
}
function ScannerDetail({ scanner, onBack, onDelete }) {
  const startScanner2 = useStartScanner();
  const stopScanner2 = useStopScanner();
  const pauseScanner2 = usePauseScanner();
  const resumeScanner2 = useResumeScanner();
  const lockScanner2 = useLockScanner();
  const unlockScanner2 = useUnlockScanner();
  const lockoutCurrent = useLockoutFrequency();
  const clearLockout2 = useClearLockout();
  const clearAllLockouts2 = useClearAllLockouts();
  const [showDeleteConfirm, setShowDeleteConfirm] = reactExports.useState(false);
  const stateColor = scanner.state === "scanning" ? "#28a745" : scanner.state === "paused" ? "#ffc107" : scanner.state === "locked" ? "#dc3545" : "#6c757d";
  return /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 3, style: { padding: "16px" }, children: [
    /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "row", justify: "between", align: "center", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "row", gap: 2, align: "center", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx(Button, { use: "secondary", size: "sm", onClick: onBack, children: "← Back" }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("h3", { style: { margin: 0, fontSize: "14px", fontWeight: 600 }, children: [
          "Scanner ",
          scanner.id
        ] })
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsx(
        "div",
        {
          style: {
            padding: "4px 10px",
            borderRadius: "12px",
            fontSize: "11px",
            fontWeight: 600,
            backgroundColor: stateColor,
            color: "white",
            textTransform: "uppercase"
          },
          children: scanner.state
        }
      )
    ] }),
    /* @__PURE__ */ jsxRuntimeExports.jsxs(
      "div",
      {
        style: {
          padding: "16px",
          backgroundColor: "#212529",
          borderRadius: "8px",
          border: "2px solid #495057"
        },
        children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("div", { style: { fontSize: "11px", color: "#adb5bd", marginBottom: "4px" }, children: "CURRENT FREQUENCY" }),
          /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { style: { fontSize: "24px", fontWeight: 700, color: "#ffffff", fontFamily: "monospace" }, children: [
            (scanner.currentFrequency / 1e6).toFixed(4),
            " MHz"
          ] }),
          /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { style: { fontSize: "11px", color: "#6c757d", marginTop: "4px" }, children: [
            scanner.currentIndex + 1,
            " / ",
            scanner.scanList.length
          ] })
        ]
      }
    ),
    /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "row", gap: 2, wrap: "wrap", children: [
      scanner.state === "stopped" && /* @__PURE__ */ jsxRuntimeExports.jsx(Button, { use: "primary", onClick: () => startScanner2.mutate(scanner.id), children: "▶ Start" }),
      scanner.state === "scanning" && /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx(Button, { use: "warning", onClick: () => pauseScanner2.mutate(scanner.id), children: "⏸ Pause" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx(Button, { use: "danger", onClick: () => stopScanner2.mutate(scanner.id), children: "⏹ Stop" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx(Button, { use: "secondary", onClick: () => lockScanner2.mutate(scanner.id), children: "🔒 Lock" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx(Button, { use: "secondary", onClick: () => lockoutCurrent.mutate(scanner.id), children: "⛔ Lockout" })
      ] }),
      scanner.state === "paused" && /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx(Button, { use: "success", onClick: () => resumeScanner2.mutate(scanner.id), children: "▶ Resume" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx(Button, { use: "danger", onClick: () => stopScanner2.mutate(scanner.id), children: "⏹ Stop" })
      ] }),
      scanner.state === "locked" && /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx(Button, { use: "success", onClick: () => unlockScanner2.mutate(scanner.id), children: "🔓 Unlock" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx(Button, { use: "danger", onClick: () => stopScanner2.mutate(scanner.id), children: "⏹ Stop" })
      ] })
    ] }),
    /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { style: { padding: "12px", backgroundColor: "#f8f9fa", borderRadius: "6px" }, children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("div", { style: { fontSize: "12px", fontWeight: 600, marginBottom: "8px" }, children: "Config" }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { style: { fontSize: "11px" }, children: [
        /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { children: [
          "Mode: ",
          /* @__PURE__ */ jsxRuntimeExports.jsx("strong", { children: scanner.mode })
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { children: [
          "Dwell: ",
          /* @__PURE__ */ jsxRuntimeExports.jsxs("strong", { children: [
            scanner.dwellTimeMs,
            "ms"
          ] })
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { children: [
          "Squelch: ",
          /* @__PURE__ */ jsxRuntimeExports.jsxs("strong", { children: [
            scanner.squelchThresholdDb,
            " dB"
          ] })
        ] })
      ] })
    ] }),
    scanner.lockoutList.length > 0 && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { style: { padding: "12px", backgroundColor: "#fff3cd", borderRadius: "6px" }, children: [
      /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "row", justify: "between", align: "center", style: { marginBottom: "8px" }, children: [
        /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { style: { fontSize: "12px", fontWeight: 600 }, children: [
          "Lockouts (",
          scanner.lockoutList.length,
          ")"
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsx(Button, { use: "warning", size: "sm", onClick: () => clearAllLockouts2.mutate(scanner.id), children: "Clear All" })
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("div", { style: { fontSize: "11px", fontFamily: "monospace" }, children: scanner.lockoutList.map((freq, idx) => /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "row", justify: "between", align: "center", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { children: [
          (freq / 1e6).toFixed(4),
          " MHz"
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsx(
          "button",
          {
            onClick: () => clearLockout2.mutate({ scannerId: scanner.id, frequency: freq }),
            style: {
              background: "none",
              border: "none",
              color: "#dc3545",
              cursor: "pointer",
              fontSize: "12px"
            },
            children: "✕"
          }
        )
      ] }, idx)) })
    ] }),
    scanner.hits.length > 0 && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { style: { padding: "12px", backgroundColor: "#d4edda", borderRadius: "6px" }, children: [
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { style: { fontSize: "12px", fontWeight: 600, marginBottom: "8px" }, children: [
        "Activity (",
        scanner.hits.length,
        ")"
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("div", { style: { maxHeight: "120px", overflowY: "auto", fontSize: "11px", fontFamily: "monospace" }, children: scanner.hits.slice().reverse().map((hit, idx) => /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { style: { padding: "3px 0" }, children: [
        new Date(hit.timestamp).toLocaleTimeString(),
        " - ",
        (hit.frequencyHz / 1e6).toFixed(4),
        " MHz"
      ] }, idx)) })
    ] }),
    /* @__PURE__ */ jsxRuntimeExports.jsx("div", { style: { marginTop: "12px", paddingTop: "12px", borderTop: "1px solid #dee2e6" }, children: !showDeleteConfirm ? /* @__PURE__ */ jsxRuntimeExports.jsx(Button, { use: "danger", size: "sm", onClick: () => setShowDeleteConfirm(true), children: "Delete" }) : /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "row", gap: 2, align: "center", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("span", { style: { fontSize: "12px", color: "#dc3545" }, children: "Are you sure?" }),
      /* @__PURE__ */ jsxRuntimeExports.jsx(Button, { use: "danger", size: "sm", onClick: onDelete, children: "Yes" }),
      /* @__PURE__ */ jsxRuntimeExports.jsx(Button, { use: "secondary", size: "sm", onClick: () => setShowDeleteConfirm(false), children: "Cancel" })
    ] }) })
  ] });
}
function CreateScannerWizard({ captureId, onCancel, onCreate }) {
  const [mode, setMode] = reactExports.useState("sequential");
  const [scanListText, setScanListText] = reactExports.useState("");
  const [dwellTimeMs, setDwellTimeMs] = reactExports.useState(500);
  const [squelchDb, setSquelchDb] = reactExports.useState(-50);
  const handleCreate = () => {
    const scanList = scanListText.split(/[,\s]+/).map((s) => parseFloat(s.trim()) * 1e6).filter((f2) => !isNaN(f2) && f2 > 0);
    if (scanList.length === 0) {
      alert("Please enter at least one frequency");
      return;
    }
    const req = {
      captureId,
      scanList,
      mode,
      dwellTimeMs,
      squelchThresholdDb: squelchDb
    };
    onCreate(req);
  };
  return /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 3, style: { padding: "16px" }, children: [
    /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "row", justify: "between", align: "center", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("h3", { style: { margin: 0, fontSize: "14px", fontWeight: 600 }, children: "Create Scanner" }),
      /* @__PURE__ */ jsxRuntimeExports.jsx(Button, { use: "secondary", size: "sm", onClick: onCancel, children: "Cancel" })
    ] }),
    /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("label", { style: { display: "block", marginBottom: "6px", fontSize: "12px", fontWeight: 600 }, children: "Frequencies (MHz, comma/space-separated)" }),
      /* @__PURE__ */ jsxRuntimeExports.jsx(
        "textarea",
        {
          value: scanListText,
          onChange: (e) => setScanListText(e.target.value),
          placeholder: "156.8 156.65 156.55 or 156.8,156.65,156.55",
          rows: 4,
          style: {
            width: "100%",
            padding: "8px",
            border: "1px solid #ced4da",
            borderRadius: "4px",
            fontSize: "11px",
            fontFamily: "monospace"
          }
        }
      )
    ] }),
    /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("label", { style: { display: "block", marginBottom: "6px", fontSize: "12px", fontWeight: 600 }, children: "Scan Mode" }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs(
        "select",
        {
          value: mode,
          onChange: (e) => setMode(e.target.value),
          style: {
            width: "100%",
            padding: "8px",
            border: "1px solid #ced4da",
            borderRadius: "4px",
            fontSize: "12px"
          },
          children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: "sequential", children: "Sequential" }),
            /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: "priority", children: "Priority" }),
            /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: "activity", children: "Activity" })
          ]
        }
      )
    ] }),
    /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "row", gap: 3, children: [
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { style: { flex: 1 }, children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("label", { style: { display: "block", marginBottom: "6px", fontSize: "12px", fontWeight: 600 }, children: "Dwell (ms)" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx(
          "input",
          {
            type: "number",
            value: dwellTimeMs,
            onChange: (e) => setDwellTimeMs(parseInt(e.target.value) || 500),
            min: 100,
            max: 5e3,
            step: 100,
            style: {
              width: "100%",
              padding: "8px",
              border: "1px solid #ced4da",
              borderRadius: "4px",
              fontSize: "12px"
            }
          }
        )
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { style: { flex: 1 }, children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("label", { style: { display: "block", marginBottom: "6px", fontSize: "12px", fontWeight: 600 }, children: "Squelch (dB)" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx(
          "input",
          {
            type: "number",
            value: squelchDb,
            onChange: (e) => setSquelchDb(parseInt(e.target.value) || -50),
            min: -80,
            max: 0,
            step: 5,
            style: {
              width: "100%",
              padding: "8px",
              border: "1px solid #ced4da",
              borderRadius: "4px",
              fontSize: "12px"
            }
          }
        )
      ] })
    ] }),
    /* @__PURE__ */ jsxRuntimeExports.jsx(Button, { use: "primary", onClick: handleCreate, children: "Create" })
  ] });
}
const gainUnits = [
  {
    name: "dB",
    multiplier: 1,
    decimals: 1,
    placeValues: [
      { label: "10", value: 10 },
      { label: "1", value: 1 },
      { label: "0.1", value: 0.1 }
    ]
  }
];
const bandwidthUnits = [
  {
    name: "kHz",
    multiplier: 1e3,
    decimals: 0,
    placeValues: [
      { label: "1000", value: 1e6 },
      { label: "100", value: 1e5 },
      { label: "10", value: 1e4 },
      { label: "1", value: 1e3 }
    ]
  },
  {
    name: "MHz",
    multiplier: 1e6,
    decimals: 3,
    placeValues: [
      { label: "1", value: 1e6 },
      { label: "0.1", value: 1e5 },
      { label: "0.01", value: 1e4 },
      { label: "0.001", value: 1e3 }
    ]
  }
];
const RadioTuner = ({ capture, device }) => {
  const { data: devices } = useDevices();
  const [showAdvanced, setShowAdvanced] = reactExports.useState(false);
  const [localDeviceId, setLocalDeviceId] = reactExports.useState(capture.deviceId);
  const [localFreq, setLocalFreq] = reactExports.useState(capture.centerHz);
  const [localGain, setLocalGain] = reactExports.useState(capture.gain ?? 0);
  const [localBandwidth, setLocalBandwidth] = reactExports.useState(capture.bandwidth ?? 2e5);
  const [localPpm, setLocalPpm] = reactExports.useState(capture.ppm ?? 0);
  const [localSampleRate, setLocalSampleRate] = reactExports.useState(capture.sampleRate);
  const [localAntenna, setLocalAntenna] = reactExports.useState(capture.antenna ?? "");
  const [localDcOffsetAuto, setLocalDcOffsetAuto] = reactExports.useState(capture.dcOffsetAuto ?? true);
  const [localIqBalanceAuto, setLocalIqBalanceAuto] = reactExports.useState(capture.iqBalanceAuto ?? true);
  const [localStreamFormat, setLocalStreamFormat] = reactExports.useState(capture.streamFormat ?? "");
  const [localElementGains, setLocalElementGains] = reactExports.useState(capture.elementGains ?? {});
  const [localDeviceSettings, setLocalDeviceSettings] = reactExports.useState(capture.deviceSettings ?? {});
  reactExports.useEffect(() => {
    setLocalDeviceId(capture.deviceId);
    setLocalFreq(capture.centerHz);
    setLocalGain(capture.gain ?? 0);
    setLocalBandwidth(capture.bandwidth ?? 2e5);
    setLocalPpm(capture.ppm ?? 0);
    setLocalSampleRate(capture.sampleRate);
    setLocalAntenna(capture.antenna ?? "");
    setLocalDcOffsetAuto(capture.dcOffsetAuto ?? true);
    setLocalIqBalanceAuto(capture.iqBalanceAuto ?? true);
    setLocalStreamFormat(capture.streamFormat ?? "");
    setLocalElementGains(capture.elementGains ?? {});
    setLocalDeviceSettings(capture.deviceSettings ?? {});
  }, [capture]);
  const debouncedFreq = useDebounce(localFreq, 100);
  const debouncedGain = useDebounce(localGain, 100);
  const debouncedBandwidth = useDebounce(localBandwidth, 100);
  const debouncedPpm = useDebounce(localPpm, 100);
  const updateMutation = useUpdateCapture();
  const startMutation = useStartCapture();
  const stopMutation = useStopCapture();
  const { data: channels } = useChannels(capture.id);
  const { getMemoryBank } = useMemoryBanks();
  const createChannel2 = useCreateChannel();
  const handleLoadMemoryBank = (bankId) => {
    const bank = getMemoryBank(bankId);
    if (!bank)
      return;
    updateMutation.mutate({
      captureId: capture.id,
      request: {
        centerHz: bank.captureConfig.centerHz,
        sampleRate: bank.captureConfig.sampleRate,
        gain: bank.captureConfig.gain ?? void 0,
        bandwidth: bank.captureConfig.bandwidth ?? void 0,
        ppm: bank.captureConfig.ppm ?? void 0,
        antenna: bank.captureConfig.antenna ?? void 0
      }
    });
    bank.channels.forEach((channelConfig) => {
      createChannel2.mutate({
        captureId: capture.id,
        request: {
          mode: channelConfig.mode,
          offsetHz: channelConfig.offsetHz,
          audioRate: channelConfig.audioRate,
          squelchDb: channelConfig.squelchDb,
          name: channelConfig.name
        }
      });
    });
  };
  reactExports.useEffect(() => {
    if (debouncedFreq !== capture.centerHz) {
      updateMutation.mutate({
        captureId: capture.id,
        request: { centerHz: debouncedFreq }
      });
    }
  }, [debouncedFreq]);
  reactExports.useEffect(() => {
    const gainValue = debouncedGain === 0 ? void 0 : debouncedGain;
    if (gainValue !== capture.gain) {
      updateMutation.mutate({
        captureId: capture.id,
        request: { gain: gainValue }
      });
    }
  }, [debouncedGain]);
  reactExports.useEffect(() => {
    if (debouncedBandwidth !== capture.bandwidth) {
      updateMutation.mutate({
        captureId: capture.id,
        request: { bandwidth: debouncedBandwidth }
      });
    }
  }, [debouncedBandwidth]);
  reactExports.useEffect(() => {
    if (debouncedPpm !== capture.ppm) {
      updateMutation.mutate({
        captureId: capture.id,
        request: { ppm: debouncedPpm }
      });
    }
  }, [debouncedPpm]);
  const handleSampleRateChange = (newRate) => {
    setLocalSampleRate(newRate);
    updateMutation.mutate({
      captureId: capture.id,
      request: { sampleRate: newRate }
    });
  };
  const handleAntennaChange = (newAntenna) => {
    setLocalAntenna(newAntenna);
    updateMutation.mutate({
      captureId: capture.id,
      request: { antenna: newAntenna }
    });
  };
  const handleDcOffsetAutoChange = (enabled) => {
    setLocalDcOffsetAuto(enabled);
    updateMutation.mutate({
      captureId: capture.id,
      request: { dcOffsetAuto: enabled }
    });
  };
  const handleIqBalanceAutoChange = (enabled) => {
    setLocalIqBalanceAuto(enabled);
    updateMutation.mutate({
      captureId: capture.id,
      request: { iqBalanceAuto: enabled }
    });
  };
  const handleStreamFormatChange = (format) => {
    setLocalStreamFormat(format);
    updateMutation.mutate({
      captureId: capture.id,
      request: { streamFormat: format || void 0 }
    });
  };
  const handleDeviceChange = (deviceId) => {
    const newDevice = devices == null ? void 0 : devices.find((d) => d.id === deviceId);
    if (!newDevice)
      return;
    setLocalDeviceId(deviceId);
    setLocalSampleRate(newDevice.sampleRates[0]);
    updateMutation.mutate({
      captureId: capture.id,
      request: {
        deviceId,
        // Reset sample rate to first available for new device
        sampleRate: newDevice.sampleRates[0]
      }
    });
  };
  const handleStartStop = () => {
    if (capture.state === "running") {
      stopMutation.mutate(capture.id);
    } else {
      startMutation.mutate(capture.id);
    }
  };
  const isRunning = capture.state === "running";
  const isFailed = capture.state === "failed";
  const isFreqPending = localFreq !== capture.centerHz || debouncedFreq !== capture.centerHz;
  const isGainPending = localGain !== (capture.gain ?? 0) || debouncedGain !== (capture.gain ?? 0);
  const isBandwidthPending = localBandwidth !== (capture.bandwidth ?? 2e5) || debouncedBandwidth !== (capture.bandwidth ?? 2e5);
  const isPpmPending = localPpm !== (capture.ppm ?? 0) || debouncedPpm !== (capture.ppm ?? 0);
  const isSampleRatePending = localSampleRate !== capture.sampleRate;
  const isAntennaPending = localAntenna !== (capture.antenna ?? "");
  const deviceFreqMin = (device == null ? void 0 : device.freqMinHz) ?? 24e6;
  const deviceFreqMax = (device == null ? void 0 : device.freqMaxHz) ?? 18e8;
  const gainMin = (device == null ? void 0 : device.gainMin) ?? 0;
  const gainMax = (device == null ? void 0 : device.gainMax) ?? 60;
  const bwMin = (device == null ? void 0 : device.bandwidthMin) ?? 2e5;
  const bwMax = (device == null ? void 0 : device.bandwidthMax) ?? 8e6;
  const ppmMin = (device == null ? void 0 : device.ppmMin) ?? -100;
  const ppmMax = (device == null ? void 0 : device.ppmMax) ?? 100;
  return /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 2, children: [
    /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "d-flex align-items-center gap-2 p-2 bg-light rounded border", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx(Radio, { size: 16, className: "flex-shrink-0" }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "fw-semibold small text-truncate", style: { maxWidth: "200px" }, children: device ? getDeviceDisplayName(device) : "Loading..." }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "small text-muted", children: [
        formatFrequencyMHz(localFreq),
        " MHz"
      ] }),
      updateMutation.isPending && /* @__PURE__ */ jsxRuntimeExports.jsx(Spinner, { size: "sm" }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: `badge ms-auto ${isFailed ? "bg-danger" : isRunning ? "bg-success" : "bg-secondary"}`, style: { fontSize: "0.7rem" }, children: capture.state.toUpperCase() })
    ] }),
    /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "card shadow-sm", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "card-header bg-body-tertiary py-1 px-2", children: /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { align: "center", gap: 1, children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx(Cpu, { size: 14 }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("small", { className: "fw-semibold mb-0", children: "Device & Control" })
      ] }) }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "card-body", style: { padding: "0.75rem" }, children: /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 2, children: [
        /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "row g-2 align-items-end", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "col-12 col-md-6", children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("label", { className: "form-label mb-1 small fw-semibold", children: "Radio Device" }),
            /* @__PURE__ */ jsxRuntimeExports.jsx(
              "select",
              {
                className: "form-select form-select-sm",
                value: localDeviceId,
                onChange: (e) => handleDeviceChange(e.target.value),
                disabled: isRunning,
                children: (devices || []).map((dev) => /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: dev.id, children: getDeviceDisplayName(dev) }, dev.id))
              }
            ),
            isRunning && /* @__PURE__ */ jsxRuntimeExports.jsx("small", { className: "text-warning d-block mt-1", style: { fontSize: "0.7rem" }, children: "Stop to change" })
          ] }),
          /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "col-12 col-md-3", children: /* @__PURE__ */ jsxRuntimeExports.jsx(
            Button,
            {
              use: isRunning ? "danger" : "success",
              size: "sm",
              onClick: handleStartStop,
              disabled: startMutation.isPending || stopMutation.isPending,
              style: { width: "100%" },
              children: isRunning ? "Stop" : "Start"
            }
          ) }),
          /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "col-12 col-md-3", children: /* @__PURE__ */ jsxRuntimeExports.jsx(
            BookmarkManager,
            {
              currentFrequency: localFreq,
              onTuneToFrequency: (freq) => setLocalFreq(freq),
              currentCapture: capture,
              currentChannels: channels,
              onLoadMemoryBank: handleLoadMemoryBank
            }
          ) })
        ] }),
        isFailed && capture.errorMessage && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "alert alert-danger mb-0 py-1 px-2", children: /* @__PURE__ */ jsxRuntimeExports.jsxs("small", { children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("strong", { children: "Error:" }),
          " ",
          capture.errorMessage
        ] }) })
      ] }) })
    ] }),
    /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "card shadow-sm", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "card-header bg-body-tertiary py-1 px-2", children: /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { align: "center", gap: 1, children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx(Settings, { size: 14 }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("small", { className: "fw-semibold mb-0", children: "Frequency Settings" })
      ] }) }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "card-body", style: { padding: "0.75rem" }, children: /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "row g-2", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "col-12", children: /* @__PURE__ */ jsxRuntimeExports.jsx(
          FrequencySelector,
          {
            label: isFreqPending ? "Frequency (updating...)" : "Frequency",
            value: localFreq,
            min: deviceFreqMin,
            max: deviceFreqMax,
            step: 1e3,
            onChange: setLocalFreq,
            info: "The center frequency your SDR will tune to. All channels are offset from this frequency."
          }
        ) }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "col-12 col-xl-6", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            NumericSelector,
            {
              label: isGainPending ? "Gain (updating...)" : "Gain",
              value: localGain,
              min: gainMin,
              max: gainMax,
              step: 0.1,
              units: gainUnits,
              info: "Signal amplification in decibels. Higher gain increases sensitivity but may introduce noise. Start around 20-30 dB and adjust for best signal-to-noise ratio.",
              onChange: setLocalGain
            }
          ),
          localGain > 45 && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "alert alert-warning py-1 px-2 mt-2 mb-0", style: { fontSize: "0.8rem" }, children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("strong", { children: "Warning:" }),
            " High gain (",
            localGain.toFixed(1),
            " dB) may cause signal clipping and distortion. Consider reducing gain to 20-40 dB for optimal performance."
          ] })
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "col-12 col-xl-6", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            NumericSelector,
            {
              label: isBandwidthPending ? "Bandwidth (updating...)" : "Bandwidth",
              value: localBandwidth,
              min: bwMin,
              max: bwMax,
              step: 1e3,
              units: bandwidthUnits,
              info: "Filter bandwidth. Wider bandwidth allows more spectrum but may include unwanted signals. Match to your signal type: FM broadcast ~200 kHz, narrowband ~10-25 kHz.",
              onChange: setLocalBandwidth
            }
          ),
          localBandwidth > localSampleRate && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "alert alert-warning py-1 px-2 mt-2 mb-0", style: { fontSize: "0.8rem" }, children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("strong", { children: "Warning:" }),
            " Bandwidth (",
            (localBandwidth / 1e6).toFixed(2),
            " MHz) is higher than sample rate (",
            (localSampleRate / 1e6).toFixed(2),
            " MHz). Bandwidth should be ≤ sample rate to avoid aliasing."
          ] }),
          localBandwidth < 15e4 && localSampleRate >= 2e5 && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "alert alert-info py-1 px-2 mt-2 mb-0", style: { fontSize: "0.8rem" }, children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("strong", { children: "Note:" }),
            " Bandwidth (",
            (localBandwidth / 1e3).toFixed(0),
            " kHz) may be too narrow for FM broadcast reception. Recommended: 150-220 kHz for WBFM, 10-25 kHz for NBFM, 10 kHz for AM."
          ] })
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "col-12 col-xl-6", children: /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 2, children: [
          /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "form-label mb-0 fw-semibold", children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx(Settings, { size: 16, className: "me-1" }),
            isSampleRatePending ? "Sample Rate (updating...)" : "Sample Rate"
          ] }),
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "select",
            {
              className: "form-select",
              value: localSampleRate,
              onChange: (e) => handleSampleRateChange(parseInt(e.target.value)),
              children: ((device == null ? void 0 : device.sampleRates) || []).map((rate) => /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: rate, children: formatSampleRate(rate) }, rate))
            }
          ),
          isRunning && /* @__PURE__ */ jsxRuntimeExports.jsx("small", { className: "text-warning", children: "Changing sample rate will briefly interrupt the stream" }),
          localSampleRate < 2e5 && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "alert alert-info py-1 px-2 mt-2 mb-0", style: { fontSize: "0.8rem" }, children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("strong", { children: "Note:" }),
            " Sample rate (",
            (localSampleRate / 1e3).toFixed(0),
            " kHz) is below 200 kHz. FM broadcast reception requires ≥200 kHz for optimal quality. Consider increasing sample rate if tuning to FM stations."
          ] })
        ] }) }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "col-12 col-xl-6", children: /* @__PURE__ */ jsxRuntimeExports.jsx(
          Slider,
          {
            label: isPpmPending ? "PPM Correction (updating...)" : "PPM Correction",
            value: localPpm,
            min: ppmMin,
            max: ppmMax,
            step: 0.1,
            coarseStep: 1,
            unit: "ppm",
            info: "Corrects frequency offset in parts-per-million caused by crystal oscillator inaccuracy. If signals appear slightly off-frequency, adjust this. Most devices need 0-5 ppm correction.",
            onChange: setLocalPpm
          }
        ) }),
        (device == null ? void 0 : device.antennas) && device.antennas.length > 0 && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "col-12", children: /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 2, children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("label", { className: "form-label mb-0 fw-semibold", children: isAntennaPending ? "Antenna (updating...)" : "Antenna" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "select",
            {
              className: "form-select",
              value: localAntenna,
              onChange: (e) => handleAntennaChange(e.target.value),
              children: device.antennas.map((ant) => /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: ant, children: ant }, ant))
            }
          ),
          isRunning && /* @__PURE__ */ jsxRuntimeExports.jsx("small", { className: "text-warning", children: "Changing antenna will briefly interrupt the stream" })
        ] }) })
      ] }) })
    ] }),
    /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "card shadow-sm", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "card-header bg-body-tertiary py-1 px-2", style: { cursor: "pointer" }, onClick: () => setShowAdvanced(!showAdvanced), children: /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { align: "center", gap: 1, children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx(Settings, { size: 14 }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("small", { className: "fw-semibold mb-0", children: "Advanced Settings" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "ms-auto", children: showAdvanced ? /* @__PURE__ */ jsxRuntimeExports.jsx(ChevronUp, { size: 14 }) : /* @__PURE__ */ jsxRuntimeExports.jsx(ChevronDown, { size: 14 }) })
      ] }) }),
      showAdvanced && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "card-body", style: { padding: "0.75rem" }, children: /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 2, children: [
        /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "form-check", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "input",
            {
              className: "form-check-input",
              type: "checkbox",
              id: "dcOffsetAuto",
              checked: localDcOffsetAuto,
              onChange: (e) => handleDcOffsetAutoChange(e.target.checked),
              disabled: isRunning
            }
          ),
          /* @__PURE__ */ jsxRuntimeExports.jsx("label", { className: "form-check-label", htmlFor: "dcOffsetAuto", children: "DC Offset Auto-Correction" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "form-text", children: "Automatically remove DC bias from IQ samples" })
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "form-check", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "input",
            {
              className: "form-check-input",
              type: "checkbox",
              id: "iqBalanceAuto",
              checked: localIqBalanceAuto,
              onChange: (e) => handleIqBalanceAutoChange(e.target.checked),
              disabled: isRunning
            }
          ),
          /* @__PURE__ */ jsxRuntimeExports.jsx("label", { className: "form-check-label", htmlFor: "iqBalanceAuto", children: "IQ Balance Auto-Correction" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "form-text", children: "Automatically correct IQ imbalance" })
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 2, children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("label", { className: "form-label mb-0 fw-semibold", children: "Stream Format" }),
          /* @__PURE__ */ jsxRuntimeExports.jsxs(
            "select",
            {
              className: "form-select form-select-sm",
              value: localStreamFormat,
              onChange: (e) => handleStreamFormatChange(e.target.value),
              disabled: isRunning,
              children: [
                /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: "", children: "Auto (CF32)" }),
                /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: "CF32", children: "CF32 (Complex Float32)" }),
                /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: "CS16", children: "CS16 (Complex Int16)" }),
                /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: "CS8", children: "CS8 (Complex Int8)" })
              ]
            }
          ),
          /* @__PURE__ */ jsxRuntimeExports.jsx("small", { className: "text-muted", children: "Stream format affects bandwidth and precision" })
        ] }),
        Object.keys(localElementGains).length > 0 && /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 2, children: [
          /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 0, children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("label", { className: "form-label mb-0 fw-semibold", children: "Element Gains" }),
            /* @__PURE__ */ jsxRuntimeExports.jsx("small", { className: "text-muted", children: "Individual gain controls for specific RF stages in your SDR. LNA (Low Noise Amplifier), IF, etc. Adjust these for fine-tuned control over different signal paths." })
          ] }),
          Object.entries(localElementGains).map(([key, value]) => /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 1, children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("label", { className: "form-label mb-0 small", children: key }),
            /* @__PURE__ */ jsxRuntimeExports.jsx(
              "input",
              {
                type: "number",
                className: "form-control form-control-sm",
                value,
                onChange: (e) => setLocalElementGains({
                  ...localElementGains,
                  [key]: parseFloat(e.target.value)
                }),
                disabled: isRunning,
                step: "0.1"
              }
            )
          ] }, key)),
          /* @__PURE__ */ jsxRuntimeExports.jsx("small", { className: "text-muted", children: "Per-element gain control (LNA, VGA, TIA, etc.)" })
        ] }),
        Object.keys(localDeviceSettings).length > 0 && /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 2, children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("label", { className: "form-label mb-0 fw-semibold", children: "Device Settings" }),
          Object.entries(localDeviceSettings).map(([key, value]) => /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 1, children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx("label", { className: "form-label mb-0 small", children: key }),
            /* @__PURE__ */ jsxRuntimeExports.jsx(
              "input",
              {
                type: "text",
                className: "form-control form-control-sm",
                value,
                onChange: (e) => setLocalDeviceSettings({
                  ...localDeviceSettings,
                  [key]: e.target.value
                }),
                disabled: isRunning
              }
            )
          ] }, key)),
          /* @__PURE__ */ jsxRuntimeExports.jsx("small", { className: "text-muted", children: "Device-specific configuration settings" })
        ] }),
        !isRunning && /* @__PURE__ */ jsxRuntimeExports.jsx("small", { className: "text-muted", children: "Start capture to apply advanced settings changes" })
      ] }) }),
      /* @__PURE__ */ jsxRuntimeExports.jsx(ScannerControl, { captureId: capture.id })
    ] })
  ] });
};
function dbfsToSUnits(dbfs) {
  const S9_DBFS = -6;
  const S1_DBFS = -54;
  if (dbfs >= S9_DBFS) {
    const overS9 = dbfs - S9_DBFS;
    return {
      sValue: 9,
      overS9Db: overS9,
      displayText: overS9 > 0 ? `S9+${overS9.toFixed(0)}` : "S9"
    };
  } else {
    const sValue = Math.max(1, Math.min(9, Math.floor((dbfs - S1_DBFS) / 6) + 1));
    return {
      sValue,
      overS9Db: 0,
      displayText: `S${sValue}`
    };
  }
}
function getSMeterColor(sValue, overS9Db) {
  if (sValue === 9 && overS9Db > 20) {
    return "#dc3545";
  } else if (sValue === 9 || sValue >= 7) {
    return "#28a745";
  } else if (sValue >= 5) {
    return "#ffc107";
  } else {
    return "#fd7e14";
  }
}
function SMeter({
  rssiDbFs,
  frequencyHz: _frequencyHz,
  width = 200,
  height = 24,
  showPeakHold: _showPeakHold = true
}) {
  if (rssiDbFs == null) {
    return /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "row", gap: 2, align: "center", style: { width: `${width}px` }, children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx(
        "div",
        {
          style: {
            flex: 1,
            height: `${height}px`,
            backgroundColor: "#e9ecef",
            borderRadius: "4px",
            border: "2px solid #dee2e6",
            display: "flex",
            alignItems: "center",
            justifyContent: "center"
          },
          children: /* @__PURE__ */ jsxRuntimeExports.jsx("span", { style: { fontSize: "10px", color: "#6c757d", fontWeight: 600 }, children: "NO SIGNAL" })
        }
      ),
      /* @__PURE__ */ jsxRuntimeExports.jsx(
        "span",
        {
          style: {
            fontSize: "11px",
            fontWeight: 700,
            minWidth: "50px",
            textAlign: "right",
            color: "#6c757d"
          },
          children: "S0"
        }
      )
    ] });
  }
  const { sValue, overS9Db, displayText } = dbfsToSUnits(rssiDbFs);
  const color = getSMeterColor(sValue, overS9Db);
  let percentage;
  if (sValue < 9) {
    percentage = (sValue - 1) / 8 * 100;
  } else {
    percentage = 100 + Math.min(overS9Db / 60 * 50, 50);
  }
  const tickMarks = [];
  for (let s = 1; s <= 9; s++) {
    const tickPosition = (s - 1) / 8 * 100;
    tickMarks.push(
      /* @__PURE__ */ jsxRuntimeExports.jsx(
        "div",
        {
          style: {
            position: "absolute",
            left: `${tickPosition}%`,
            top: 0,
            bottom: 0,
            width: "1px",
            backgroundColor: s % 2 === 1 ? "#adb5bd" : "#dee2e6",
            opacity: 0.6
          }
        },
        s
      )
    );
  }
  return /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "row", gap: 2, align: "center", style: { width: `${width}px` }, children: [
    /* @__PURE__ */ jsxRuntimeExports.jsxs(
      "div",
      {
        style: {
          position: "relative",
          flex: 1,
          height: `${height}px`,
          backgroundColor: "#212529",
          borderRadius: "4px",
          overflow: "hidden",
          border: "2px solid #495057",
          boxShadow: "inset 0 2px 4px rgba(0,0,0,0.3)"
        },
        children: [
          tickMarks,
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "div",
            {
              style: {
                position: "absolute",
                left: 0,
                top: 0,
                height: "100%",
                width: "50%",
                background: "linear-gradient(to right, rgba(253, 126, 20, 0.15), rgba(255, 193, 7, 0.15))"
              }
            }
          ),
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "div",
            {
              style: {
                position: "absolute",
                left: "50%",
                top: 0,
                height: "100%",
                width: "37.5%",
                background: "linear-gradient(to right, rgba(255, 193, 7, 0.15), rgba(40, 167, 69, 0.15))"
              }
            }
          ),
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "div",
            {
              style: {
                position: "absolute",
                left: "87.5%",
                top: 0,
                height: "100%",
                width: "12.5%",
                background: "rgba(40, 167, 69, 0.2)"
              }
            }
          ),
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "div",
            {
              style: {
                position: "absolute",
                left: 0,
                right: 0,
                top: "2px",
                display: "flex",
                justifyContent: "space-between",
                padding: "0 4px",
                pointerEvents: "none"
              },
              children: [1, 3, 5, 7, 9].map((s) => /* @__PURE__ */ jsxRuntimeExports.jsx(
                "span",
                {
                  style: {
                    fontSize: "8px",
                    color: "#adb5bd",
                    fontWeight: 600,
                    textShadow: "0 1px 2px rgba(0,0,0,0.8)"
                  },
                  children: s
                },
                s
              ))
            }
          ),
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "div",
            {
              style: {
                position: "absolute",
                left: 0,
                top: 0,
                height: "100%",
                width: `${Math.min(percentage, 100)}%`,
                backgroundColor: color,
                transition: "width 0.15s ease-out, background-color 0.15s ease-out",
                boxShadow: percentage > 0 ? `0 0 8px ${color}` : "none",
                borderRight: percentage > 0 && percentage < 100 ? `2px solid ${color}` : "none"
              }
            }
          ),
          percentage > 100 && /* @__PURE__ */ jsxRuntimeExports.jsx(
            "div",
            {
              style: {
                position: "absolute",
                right: 0,
                top: 0,
                height: "100%",
                width: `${percentage - 100}%`,
                backgroundColor: "#dc3545",
                boxShadow: "0 0 10px #dc3545",
                opacity: 0.8
              }
            }
          )
        ]
      }
    ),
    /* @__PURE__ */ jsxRuntimeExports.jsx(
      "span",
      {
        style: {
          fontSize: "12px",
          fontWeight: 700,
          minWidth: "55px",
          textAlign: "right",
          color,
          textShadow: "0 1px 2px rgba(0,0,0,0.3)",
          fontFamily: "monospace"
        },
        children: displayText
      }
    )
  ] });
}
const API_BASE$1 = "/api/v1";
async function fetchFrequencyName(frequencyHz) {
  const response = await fetch(
    `${API_BASE$1}/frequency/identify?frequency_hz=${frequencyHz}`
  );
  if (!response.ok) {
    return null;
  }
  const data = await response.json();
  return (data == null ? void 0 : data.name) || null;
}
const FrequencyLabel = ({ frequencyHz, autoName }) => {
  const { data: fetchedName } = useQuery({
    queryKey: ["frequencyName", frequencyHz],
    queryFn: () => fetchFrequencyName(frequencyHz),
    enabled: !autoName,
    // Only fetch if autoName not provided
    staleTime: Infinity
    // Frequency names don't change
  });
  const displayName = autoName || fetchedName;
  if (!displayName) {
    return null;
  }
  return /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: displayName });
};
function formatChannelId(id2) {
  const match = id2.match(/^ch(\d+)$/);
  return match ? `Ch ${match[1]}` : id2;
}
const CompactChannelCard = ({
  channel,
  capture,
  isPlaying,
  onTogglePlay,
  onCopyUrl,
  copiedUrl
}) => {
  var _a2, _b2;
  const [isExpanded, setIsExpanded] = reactExports.useState(false);
  const [showStreamDropdown, setShowStreamDropdown] = reactExports.useState(false);
  const [isEditingName, setIsEditingName] = reactExports.useState(false);
  const [editNameValue, setEditNameValue] = reactExports.useState("");
  const [newNotchFreq, setNewNotchFreq] = reactExports.useState("");
  const [showDspFilters, setShowDspFilters] = reactExports.useState(false);
  const [showAgcSettings, setShowAgcSettings] = reactExports.useState(false);
  const [showNoiseBlanker, setShowNoiseBlanker] = reactExports.useState(false);
  const nameInputRef = reactExports.useRef(null);
  const updateChannel2 = useUpdateChannel(capture.id);
  const deleteChannel2 = useDeleteChannel();
  const toast = useToast();
  const getChannelFrequency = () => capture.centerHz + channel.offsetHz;
  const displayName = channel.name || channel.autoName || formatChannelId(channel.id);
  reactExports.useEffect(() => {
    if (isEditingName && nameInputRef.current) {
      nameInputRef.current.focus();
      nameInputRef.current.select();
    }
  }, [isEditingName]);
  const handleStartEditName = () => {
    setEditNameValue(channel.name || channel.autoName || "");
    setIsEditingName(true);
  };
  const handleSaveEditName = () => {
    const trimmedValue = editNameValue.trim();
    updateChannel2.mutate({
      channelId: channel.id,
      request: { name: trimmedValue || null }
    }, {
      onSuccess: () => {
        toast.success("Channel name updated");
      },
      onError: (error) => {
        toast.error((error == null ? void 0 : error.message) || "Failed to update channel name");
      }
    });
    setIsEditingName(false);
  };
  const handleKeyDown = (e) => {
    if (e.key === "Enter") {
      handleSaveEditName();
    } else if (e.key === "Escape") {
      setIsEditingName(false);
    }
  };
  const handleDelete = () => {
    if (confirm("Delete this channel?")) {
      deleteChannel2.mutate(channel.id, {
        onSuccess: () => {
          toast.success("Channel deleted successfully");
        },
        onError: (error) => {
          toast.error((error == null ? void 0 : error.message) || "Failed to delete channel");
        }
      });
    }
  };
  const updateChannelWithToast = (request) => {
    updateChannel2.mutate({
      channelId: channel.id,
      request
    }, {
      onError: (error) => {
        toast.error((error == null ? void 0 : error.message) || "Failed to update channel");
      }
    });
  };
  const handleAddNotch = () => {
    const freq = parseFloat(newNotchFreq);
    if (isNaN(freq) || freq <= 0 || freq > 2e4) {
      toast.error("Notch frequency must be between 0 and 20000 Hz");
      return;
    }
    const currentNotches = channel.notchFrequencies || [];
    if (currentNotches.includes(freq)) {
      toast.error("This frequency is already in the notch list");
      return;
    }
    if (currentNotches.length >= 10) {
      toast.error("Maximum 10 notch filters allowed");
      return;
    }
    updateChannelWithToast({ notchFrequencies: [...currentNotches, freq] });
    setNewNotchFreq("");
    toast.success(`Added notch filter at ${freq} Hz`);
  };
  const handleRemoveNotch = (freq) => {
    const currentNotches = channel.notchFrequencies || [];
    updateChannelWithToast({
      notchFrequencies: currentNotches.filter((f2) => f2 !== freq)
    });
    toast.success(`Removed notch filter at ${freq} Hz`);
  };
  const streamFormats = [
    { format: "PCM", ext: ".pcm", label: "Raw PCM" },
    { format: "MP3", ext: ".mp3", label: "MP3 (128k)" },
    { format: "Opus", ext: ".opus", label: "Opus" },
    { format: "AAC", ext: ".aac", label: "AAC" }
  ];
  return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "card shadow-sm h-100", children: [
    /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "card-header bg-body-tertiary p-2", children: /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { justify: "between", align: "center", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "small fw-semibold text-truncate", style: { flex: 1, minWidth: 0 }, children: [
        formatChannelId(channel.id),
        " • ",
        formatFrequencyMHz(getChannelFrequency()),
        " MHz"
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { gap: 1, children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx(
          Button,
          {
            use: isPlaying ? "warning" : "success",
            size: "sm",
            onClick: onTogglePlay,
            title: isPlaying ? "Stop Listening" : "Listen Now",
            className: "px-2 py-1",
            children: /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { align: "center", gap: 1, children: [
              isPlaying ? /* @__PURE__ */ jsxRuntimeExports.jsx(VolumeX, { size: 14 }) : /* @__PURE__ */ jsxRuntimeExports.jsx(Volume2, { size: 14 }),
              /* @__PURE__ */ jsxRuntimeExports.jsx("span", { style: { fontSize: "11px" }, children: isPlaying ? "Stop" : "Listen" })
            ] })
          }
        ),
        /* @__PURE__ */ jsxRuntimeExports.jsx(
          Button,
          {
            use: "secondary",
            size: "sm",
            onClick: () => setIsExpanded(!isExpanded),
            title: isExpanded ? "Collapse" : "Settings",
            className: "p-1",
            children: isExpanded ? /* @__PURE__ */ jsxRuntimeExports.jsx(ChevronUp, { size: 14 }) : /* @__PURE__ */ jsxRuntimeExports.jsx(Settings, { size: 14 })
          }
        ),
        /* @__PURE__ */ jsxRuntimeExports.jsx(
          Button,
          {
            use: "danger",
            size: "sm",
            appearance: "outline",
            onClick: handleDelete,
            title: "Delete",
            className: "p-1",
            children: /* @__PURE__ */ jsxRuntimeExports.jsx(Trash2, { size: 14 })
          }
        )
      ] })
    ] }) }),
    /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "card-body p-2", children: /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 2, children: [
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { children: [
        /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { justify: "between", align: "center", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "small text-muted", children: channel.mode.toUpperCase() }),
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "button",
            {
              className: "btn btn-sm p-0",
              style: { width: "16px", height: "16px" },
              onClick: handleStartEditName,
              title: "Edit name",
              children: /* @__PURE__ */ jsxRuntimeExports.jsx(Pen, { size: 12 })
            }
          )
        ] }),
        isEditingName ? /* @__PURE__ */ jsxRuntimeExports.jsx(
          "input",
          {
            ref: nameInputRef,
            type: "text",
            className: "form-control form-control-sm mt-1",
            value: editNameValue,
            onChange: (e) => setEditNameValue(e.target.value),
            onBlur: handleSaveEditName,
            onKeyDown: handleKeyDown,
            placeholder: "Channel name"
          }
        ) : /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "fw-semibold", title: channel.name && channel.autoName ? `Auto: ${channel.autoName}` : void 0, children: displayName }),
          /* @__PURE__ */ jsxRuntimeExports.jsx(FrequencyLabel, { frequencyHz: getChannelFrequency(), autoName: channel.name ? channel.autoName : null }),
          (() => {
            const channelFreq = getChannelFrequency();
            const spectrumMin = capture.centerHz - capture.sampleRate / 2;
            const spectrumMax = capture.centerHz + capture.sampleRate / 2;
            const isOutOfRange = channelFreq < spectrumMin || channelFreq > spectrumMax;
            if (isOutOfRange) {
              return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "alert alert-warning py-1 px-2 mt-1 mb-0", style: { fontSize: "0.7rem" }, children: [
                /* @__PURE__ */ jsxRuntimeExports.jsx("strong", { children: "Warning:" }),
                " Channel frequency (",
                formatFrequencyMHz(channelFreq),
                " MHz) is outside observable spectrum (",
                formatFrequencyMHz(spectrumMin),
                " - ",
                formatFrequencyMHz(spectrumMax),
                " MHz)"
              ] });
            }
            return null;
          })()
        ] })
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "border rounded p-2 bg-light", children: /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 1, children: [
        /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { align: "center", gap: 1, children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "badge bg-success text-white", style: { fontSize: "8px", width: "32px" }, children: "LIVE" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx("div", { style: { flex: 1 }, children: /* @__PURE__ */ jsxRuntimeExports.jsx(SMeter, { rssiDbFs: channel.rssiDb, frequencyHz: getChannelFrequency(), width: 200, height: 24 }) })
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "row", gap: 1, align: "center", style: { fontSize: "9px" }, children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "text-muted", style: { width: "40px" }, children: "RSSI:" }),
          /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "fw-semibold", children: [
            ((_a2 = channel.rssiDb) == null ? void 0 : _a2.toFixed(1)) ?? "N/A",
            " dBFS"
          ] }),
          /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "text-muted ms-2", style: { width: "35px" }, children: "SNR:" }),
          /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "fw-semibold", children: [
            ((_b2 = channel.snrDb) == null ? void 0 : _b2.toFixed(1)) ?? "N/A",
            " dB"
          ] })
        ] })
      ] }) }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "dropdown", style: { position: "relative" }, children: [
        /* @__PURE__ */ jsxRuntimeExports.jsxs(
          Button,
          {
            use: "secondary",
            size: "sm",
            onClick: () => setShowStreamDropdown(!showStreamDropdown),
            className: "w-100 d-flex justify-content-between align-items-center",
            children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "small", children: "Copy Stream URL" }),
              /* @__PURE__ */ jsxRuntimeExports.jsx(ChevronDown, { size: 12 })
            ]
          }
        ),
        showStreamDropdown && /* @__PURE__ */ jsxRuntimeExports.jsx(
          "div",
          {
            className: "dropdown-menu show w-100",
            style: { position: "absolute", top: "100%", zIndex: 1e3 },
            children: streamFormats.map(({ format, ext, label }) => {
              const url = `${window.location.origin}/api/v1/stream/channels/${channel.id}${ext}`;
              const isCopied = copiedUrl === url;
              return /* @__PURE__ */ jsxRuntimeExports.jsxs(
                "button",
                {
                  className: "dropdown-item d-flex justify-content-between align-items-center",
                  onClick: () => {
                    onCopyUrl(url);
                    setShowStreamDropdown(false);
                  },
                  children: [
                    /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "small", children: label }),
                    isCopied ? /* @__PURE__ */ jsxRuntimeExports.jsx(CheckCircle, { size: 12, className: "text-success" }) : /* @__PURE__ */ jsxRuntimeExports.jsx(Copy, { size: 12 })
                  ]
                },
                format
              );
            })
          }
        )
      ] }),
      isExpanded && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "border-top pt-2 mt-1", children: /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 2, children: [
        /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 1, children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("label", { className: "form-label small mb-0", children: "Mode" }),
          /* @__PURE__ */ jsxRuntimeExports.jsxs(
            "select",
            {
              className: "form-select form-select-sm",
              value: channel.mode,
              onChange: (e) => updateChannelWithToast({ mode: e.target.value }),
              children: [
                /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: "wbfm", children: "WBFM" }),
                /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: "nbfm", children: "NBFM" }),
                /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: "am", children: "AM" }),
                /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: "ssb", children: "SSB" }),
                /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: "raw", children: "Raw IQ" }),
                /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: "p25", children: "P25" }),
                /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: "dmr", children: "DMR" })
              ]
            }
          )
        ] }),
        channel.mode === "ssb" && /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 1, children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("label", { className: "form-label small mb-0", children: "SSB Mode" }),
          /* @__PURE__ */ jsxRuntimeExports.jsxs(
            "select",
            {
              className: "form-select form-select-sm",
              value: channel.ssbMode,
              onChange: (e) => updateChannelWithToast({ ssbMode: e.target.value }),
              children: [
                /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: "usb", children: "USB (Upper Sideband)" }),
                /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: "lsb", children: "LSB (Lower Sideband)" })
              ]
            }
          ),
          /* @__PURE__ */ jsxRuntimeExports.jsx("small", { className: "text-muted", children: "USB: Amateur radio above 10 MHz. LSB: Below 10 MHz" })
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsx(
          Slider,
          {
            label: "Squelch",
            value: channel.squelchDb ?? -60,
            min: -80,
            max: 0,
            step: 1,
            coarseStep: 10,
            unit: "dB",
            formatValue: (val) => `${val.toFixed(0)} dB`,
            onChange: (val) => updateChannelWithToast({ squelchDb: val })
          }
        ),
        /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 1, children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("label", { className: "form-label small mb-0", children: "Audio Rate" }),
          /* @__PURE__ */ jsxRuntimeExports.jsxs(
            "select",
            {
              className: "form-select form-select-sm",
              value: channel.audioRate,
              onChange: (e) => updateChannelWithToast({ audioRate: parseInt(e.target.value) }),
              children: [
                /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: 8e3, children: "8 kHz" }),
                /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: 16e3, children: "16 kHz" }),
                /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: 24e3, children: "24 kHz" }),
                /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: 48e3, children: "48 kHz" })
              ]
            }
          )
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 1, children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            FrequencySelector,
            {
              label: "Frequency",
              value: getChannelFrequency(),
              min: capture.centerHz - capture.sampleRate / 2,
              max: capture.centerHz + capture.sampleRate / 2,
              step: 1e3,
              onChange: (hz) => updateChannelWithToast({ offsetHz: hz - capture.centerHz })
            }
          ),
          /* @__PURE__ */ jsxRuntimeExports.jsxs("small", { className: "text-muted", children: [
            "Offset: ",
            (channel.offsetHz / 1e3).toFixed(0),
            " kHz"
          ] })
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "border rounded", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsxs(
            "button",
            {
              className: "btn btn-sm w-100 text-start d-flex justify-content-between align-items-center p-2",
              onClick: () => setShowDspFilters(!showDspFilters),
              style: { background: "transparent", border: "none" },
              children: [
                /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "fw-semibold small", children: "DSP Filters" }),
                showDspFilters ? /* @__PURE__ */ jsxRuntimeExports.jsx(ChevronUp, { size: 14 }) : /* @__PURE__ */ jsxRuntimeExports.jsx(ChevronDown, { size: 14 })
              ]
            }
          ),
          showDspFilters && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "p-2 border-top", children: /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 2, children: [
            /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "alert alert-info py-1 px-2 mb-0", style: { fontSize: "0.7rem" }, children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx("strong", { children: "DSP Filters" }),
              " shape audio frequency response for optimal clarity"
            ] }),
            (channel.mode === "wbfm" || channel.mode === "nbfm") && /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
              /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 1, children: [
                /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { justify: "between", align: "center", children: [
                  /* @__PURE__ */ jsxRuntimeExports.jsx("label", { className: "form-label small mb-0", children: "Deemphasis" }),
                  /* @__PURE__ */ jsxRuntimeExports.jsx(
                    "input",
                    {
                      type: "checkbox",
                      checked: channel.enableDeemphasis,
                      onChange: (e) => updateChannelWithToast({ enableDeemphasis: e.target.checked }),
                      style: { width: "16px", height: "16px" }
                    }
                  )
                ] }),
                channel.enableDeemphasis && /* @__PURE__ */ jsxRuntimeExports.jsxs(
                  "select",
                  {
                    className: "form-select form-select-sm",
                    value: channel.deemphasisTauUs,
                    onChange: (e) => updateChannelWithToast({ deemphasisTauUs: parseFloat(e.target.value) }),
                    children: [
                      /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: 50, children: "50 µs (Europe)" }),
                      /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: 75, children: "75 µs (USA)" })
                    ]
                  }
                ),
                /* @__PURE__ */ jsxRuntimeExports.jsx("small", { className: "text-muted", children: "Compensates for FM pre-emphasis (boosts treble)" })
              ] }),
              channel.mode === "wbfm" && /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 1, children: [
                /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { justify: "between", align: "center", children: [
                  /* @__PURE__ */ jsxRuntimeExports.jsx("label", { className: "form-label small mb-0", children: "MPX Filter (19 kHz Pilot Removal)" }),
                  /* @__PURE__ */ jsxRuntimeExports.jsx(
                    "input",
                    {
                      type: "checkbox",
                      checked: channel.enableMpxFilter,
                      onChange: (e) => updateChannelWithToast({ enableMpxFilter: e.target.checked }),
                      style: { width: "16px", height: "16px" }
                    }
                  )
                ] }),
                /* @__PURE__ */ jsxRuntimeExports.jsx("small", { className: "text-muted", children: "Removes stereo pilot tone and subcarriers (eliminates high-pitch whine)" })
              ] }),
              /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 1, children: [
                /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { justify: "between", align: "center", children: [
                  /* @__PURE__ */ jsxRuntimeExports.jsx("label", { className: "form-label small mb-0", children: "Highpass Filter" }),
                  /* @__PURE__ */ jsxRuntimeExports.jsx(
                    "input",
                    {
                      type: "checkbox",
                      checked: channel.enableFmHighpass,
                      onChange: (e) => updateChannelWithToast({ enableFmHighpass: e.target.checked }),
                      style: { width: "16px", height: "16px" }
                    }
                  )
                ] }),
                channel.enableFmHighpass && /* @__PURE__ */ jsxRuntimeExports.jsx(
                  Slider,
                  {
                    label: "",
                    value: channel.fmHighpassHz,
                    min: 50,
                    max: 500,
                    step: 10,
                    unit: "Hz",
                    formatValue: (val) => `${val.toFixed(0)} Hz`,
                    onChange: (val) => updateChannelWithToast({ fmHighpassHz: val })
                  }
                ),
                /* @__PURE__ */ jsxRuntimeExports.jsx("small", { className: "text-muted", children: "Removes DC offset and rumble" })
              ] }),
              channel.mode === "nbfm" && /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 1, children: [
                /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { justify: "between", align: "center", children: [
                  /* @__PURE__ */ jsxRuntimeExports.jsx("label", { className: "form-label small mb-0", children: "Lowpass Filter (Voice BW)" }),
                  /* @__PURE__ */ jsxRuntimeExports.jsx(
                    "input",
                    {
                      type: "checkbox",
                      checked: channel.enableFmLowpass,
                      onChange: (e) => updateChannelWithToast({ enableFmLowpass: e.target.checked }),
                      style: { width: "16px", height: "16px" }
                    }
                  )
                ] }),
                channel.enableFmLowpass && /* @__PURE__ */ jsxRuntimeExports.jsx(
                  Slider,
                  {
                    label: "",
                    value: channel.fmLowpassHz,
                    min: 2e3,
                    max: 5e3,
                    step: 100,
                    unit: "Hz",
                    formatValue: (val) => `${val.toFixed(0)} Hz`,
                    onChange: (val) => updateChannelWithToast({ fmLowpassHz: val })
                  }
                ),
                /* @__PURE__ */ jsxRuntimeExports.jsx("small", { className: "text-muted", children: "Limits voice bandwidth (3000 Hz typical)" })
              ] })
            ] }),
            channel.mode === "am" && /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
              /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 1, children: [
                /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { justify: "between", align: "center", children: [
                  /* @__PURE__ */ jsxRuntimeExports.jsx("label", { className: "form-label small mb-0", children: "Highpass Filter (DC Removal)" }),
                  /* @__PURE__ */ jsxRuntimeExports.jsx(
                    "input",
                    {
                      type: "checkbox",
                      checked: channel.enableAmHighpass,
                      onChange: (e) => updateChannelWithToast({ enableAmHighpass: e.target.checked }),
                      style: { width: "16px", height: "16px" }
                    }
                  )
                ] }),
                channel.enableAmHighpass && /* @__PURE__ */ jsxRuntimeExports.jsx(
                  Slider,
                  {
                    label: "",
                    value: channel.amHighpassHz,
                    min: 50,
                    max: 500,
                    step: 10,
                    unit: "Hz",
                    formatValue: (val) => `${val.toFixed(0)} Hz`,
                    onChange: (val) => updateChannelWithToast({ amHighpassHz: val })
                  }
                ),
                /* @__PURE__ */ jsxRuntimeExports.jsx("small", { className: "text-muted", children: "Removes AM carrier offset and rumble" })
              ] }),
              /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 1, children: [
                /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { justify: "between", align: "center", children: [
                  /* @__PURE__ */ jsxRuntimeExports.jsx("label", { className: "form-label small mb-0", children: "Lowpass Filter (Bandwidth)" }),
                  /* @__PURE__ */ jsxRuntimeExports.jsx(
                    "input",
                    {
                      type: "checkbox",
                      checked: channel.enableAmLowpass,
                      onChange: (e) => updateChannelWithToast({ enableAmLowpass: e.target.checked }),
                      style: { width: "16px", height: "16px" }
                    }
                  )
                ] }),
                channel.enableAmLowpass && /* @__PURE__ */ jsxRuntimeExports.jsx(
                  Slider,
                  {
                    label: "",
                    value: channel.amLowpassHz,
                    min: 3e3,
                    max: 1e4,
                    step: 100,
                    unit: "Hz",
                    formatValue: (val) => `${val.toFixed(0)} Hz`,
                    onChange: (val) => updateChannelWithToast({ amLowpassHz: val })
                  }
                ),
                /* @__PURE__ */ jsxRuntimeExports.jsx("small", { className: "text-muted", children: "AM broadcast: 5000 Hz, Aviation: 3000 Hz" })
              ] })
            ] }),
            channel.mode === "ssb" && /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 1, children: [
              /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { justify: "between", align: "center", children: [
                /* @__PURE__ */ jsxRuntimeExports.jsx("label", { className: "form-label small mb-0", children: "Bandpass Filter (Voice)" }),
                /* @__PURE__ */ jsxRuntimeExports.jsx(
                  "input",
                  {
                    type: "checkbox",
                    checked: channel.enableSsbBandpass,
                    onChange: (e) => updateChannelWithToast({ enableSsbBandpass: e.target.checked }),
                    style: { width: "16px", height: "16px" }
                  }
                )
              ] }),
              channel.enableSsbBandpass && /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 1, children: [
                /* @__PURE__ */ jsxRuntimeExports.jsx(
                  Slider,
                  {
                    label: "Low Cutoff",
                    value: channel.ssbBandpassLowHz,
                    min: 100,
                    max: 1e3,
                    step: 50,
                    unit: "Hz",
                    formatValue: (val) => `${val.toFixed(0)} Hz`,
                    onChange: (val) => updateChannelWithToast({ ssbBandpassLowHz: val })
                  }
                ),
                /* @__PURE__ */ jsxRuntimeExports.jsx(
                  Slider,
                  {
                    label: "High Cutoff",
                    value: channel.ssbBandpassHighHz,
                    min: 2e3,
                    max: 4e3,
                    step: 100,
                    unit: "Hz",
                    formatValue: (val) => `${val.toFixed(0)} Hz`,
                    onChange: (val) => updateChannelWithToast({ ssbBandpassHighHz: val })
                  }
                )
              ] }),
              /* @__PURE__ */ jsxRuntimeExports.jsx("small", { className: "text-muted", children: "Typical voice: 300-3000 Hz. Narrow: 500-2500 Hz" })
            ] })
          ] }) })
        ] }),
        (channel.mode === "am" || channel.mode === "ssb") && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "border rounded", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsxs(
            "button",
            {
              className: "btn btn-sm w-100 text-start d-flex justify-content-between align-items-center p-2",
              onClick: () => setShowAgcSettings(!showAgcSettings),
              style: { background: "transparent", border: "none" },
              children: [
                /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { align: "center", gap: 1, children: [
                  /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "fw-semibold small", children: "AGC (Auto Gain Control)" }),
                  /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: `badge ${channel.enableAgc ? "bg-success" : "bg-secondary"}`, style: { fontSize: "8px" }, children: channel.enableAgc ? "ON" : "OFF" })
                ] }),
                showAgcSettings ? /* @__PURE__ */ jsxRuntimeExports.jsx(ChevronUp, { size: 14 }) : /* @__PURE__ */ jsxRuntimeExports.jsx(ChevronDown, { size: 14 })
              ]
            }
          ),
          showAgcSettings && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "p-2 border-top", children: /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 2, children: [
            /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "alert alert-info py-1 px-2 mb-0", style: { fontSize: "0.7rem" }, children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx("strong", { children: "AGC" }),
              " automatically adjusts gain to maintain consistent audio levels. Essential for AM/SSB signals with varying strength."
            ] }),
            /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { justify: "between", align: "center", children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx("label", { className: "form-label small mb-0", children: "Enable AGC" }),
              /* @__PURE__ */ jsxRuntimeExports.jsx(
                "input",
                {
                  type: "checkbox",
                  checked: channel.enableAgc,
                  onChange: (e) => updateChannelWithToast({ enableAgc: e.target.checked }),
                  style: { width: "16px", height: "16px" }
                }
              )
            ] }),
            channel.enableAgc && /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx(
                Slider,
                {
                  label: "Target Level",
                  value: channel.agcTargetDb,
                  min: -60,
                  max: -10,
                  step: 1,
                  unit: "dB",
                  formatValue: (val) => `${val.toFixed(0)} dB`,
                  onChange: (val) => updateChannelWithToast({ agcTargetDb: val })
                }
              ),
              /* @__PURE__ */ jsxRuntimeExports.jsx(
                Slider,
                {
                  label: "Attack Time",
                  value: channel.agcAttackMs,
                  min: 1,
                  max: 100,
                  step: 1,
                  unit: "ms",
                  formatValue: (val) => `${val.toFixed(0)} ms`,
                  onChange: (val) => updateChannelWithToast({ agcAttackMs: val })
                }
              ),
              /* @__PURE__ */ jsxRuntimeExports.jsx(
                Slider,
                {
                  label: "Release Time",
                  value: channel.agcReleaseMs,
                  min: 10,
                  max: 500,
                  step: 10,
                  unit: "ms",
                  formatValue: (val) => `${val.toFixed(0)} ms`,
                  onChange: (val) => updateChannelWithToast({ agcReleaseMs: val })
                }
              ),
              /* @__PURE__ */ jsxRuntimeExports.jsx("small", { className: "text-muted", children: "Attack: how quickly gain increases. Release: how quickly gain decreases." })
            ] })
          ] }) })
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "border rounded", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsxs(
            "button",
            {
              className: "btn btn-sm w-100 text-start d-flex justify-content-between align-items-center p-2",
              onClick: () => setShowNoiseBlanker(!showNoiseBlanker),
              style: { background: "transparent", border: "none" },
              children: [
                /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { align: "center", gap: 1, children: [
                  /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "fw-semibold small", children: "Noise Blanker" }),
                  /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: `badge ${channel.enableNoiseBlanker ? "bg-success" : "bg-secondary"}`, style: { fontSize: "8px" }, children: channel.enableNoiseBlanker ? "ON" : "OFF" })
                ] }),
                showNoiseBlanker ? /* @__PURE__ */ jsxRuntimeExports.jsx(ChevronUp, { size: 14 }) : /* @__PURE__ */ jsxRuntimeExports.jsx(ChevronDown, { size: 14 })
              ]
            }
          ),
          showNoiseBlanker && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "p-2 border-top", children: /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 2, children: [
            /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "alert alert-warning py-1 px-2 mb-0", style: { fontSize: "0.7rem" }, children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx("strong", { children: "Noise Blanker" }),
              " suppresses impulse noise from lightning, ignition, power lines. Use when experiencing static pops/clicks."
            ] }),
            /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { justify: "between", align: "center", children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx("label", { className: "form-label small mb-0", children: "Enable Noise Blanker" }),
              /* @__PURE__ */ jsxRuntimeExports.jsx(
                "input",
                {
                  type: "checkbox",
                  checked: channel.enableNoiseBlanker,
                  onChange: (e) => updateChannelWithToast({ enableNoiseBlanker: e.target.checked }),
                  style: { width: "16px", height: "16px" }
                }
              )
            ] }),
            channel.enableNoiseBlanker && /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx(
                Slider,
                {
                  label: "Threshold",
                  value: channel.noiseBlankerThresholdDb,
                  min: 3,
                  max: 30,
                  step: 1,
                  unit: "dB",
                  formatValue: (val) => `${val.toFixed(0)} dB`,
                  onChange: (val) => updateChannelWithToast({ noiseBlankerThresholdDb: val })
                }
              ),
              /* @__PURE__ */ jsxRuntimeExports.jsx("small", { className: "text-muted", children: "Lower = more aggressive (may remove weak signals). Higher = less aggressive. Start at 10 dB." })
            ] })
          ] }) })
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 1, children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("label", { className: "form-label small mb-0", children: "Notch Filters (Interference Rejection)" }),
          channel.notchFrequencies && channel.notchFrequencies.length > 0 ? /* @__PURE__ */ jsxRuntimeExports.jsx(Flex, { direction: "column", gap: 1, children: channel.notchFrequencies.map((freq) => /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { justify: "between", align: "center", className: "border rounded p-1 bg-light", children: [
            /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "small fw-semibold", children: [
              freq,
              " Hz"
            ] }),
            /* @__PURE__ */ jsxRuntimeExports.jsx(
              "button",
              {
                className: "btn btn-sm btn-danger p-0",
                style: { width: "20px", height: "20px", fontSize: "12px" },
                onClick: () => handleRemoveNotch(freq),
                title: "Remove notch",
                children: "×"
              }
            )
          ] }, freq)) }) : /* @__PURE__ */ jsxRuntimeExports.jsx("small", { className: "text-muted", children: "No notch filters active" }),
          /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { gap: 1, children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx(
              "input",
              {
                type: "number",
                className: "form-control form-control-sm",
                placeholder: "Frequency (Hz)",
                value: newNotchFreq,
                onChange: (e) => setNewNotchFreq(e.target.value),
                onKeyDown: (e) => e.key === "Enter" && handleAddNotch(),
                min: 0,
                max: 2e4,
                step: 10
              }
            ),
            /* @__PURE__ */ jsxRuntimeExports.jsx(
              Button,
              {
                use: "primary",
                size: "sm",
                onClick: handleAddNotch,
                disabled: !newNotchFreq,
                children: "Add"
              }
            )
          ] }),
          /* @__PURE__ */ jsxRuntimeExports.jsx("small", { className: "text-muted", children: "Remove interfering tones (power line hum, carriers, etc.). Common: 60 Hz, 120 Hz" })
        ] })
      ] }) })
    ] }) }),
    /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "card-footer p-1 bg-body-tertiary", children: /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { justify: "between", align: "center", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: `badge bg-${channel.state === "running" ? "success" : "secondary"}`, style: { fontSize: "9px" }, children: channel.state }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("small", { className: "text-muted", style: { fontSize: "9px" }, children: [
        (channel.offsetHz / 1e3).toFixed(0),
        " kHz • ",
        channel.audioRate / 1e3,
        " kHz"
      ] })
    ] }) })
  ] });
};
const Skeleton = ({
  width = "100%",
  height = "20px",
  borderRadius = "4px",
  className = ""
}) => {
  return /* @__PURE__ */ jsxRuntimeExports.jsx(
    "div",
    {
      className: `skeleton ${className}`,
      style: {
        width,
        height,
        borderRadius,
        backgroundColor: "#e9ecef",
        animation: "pulse 1.5s ease-in-out infinite"
      }
    }
  );
};
const SkeletonChannelCard = () => {
  return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "card shadow-sm h-100", children: [
    /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "card-header bg-body-tertiary p-2", children: /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "d-flex justify-content-between align-items-center", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx(Skeleton, { width: "120px", height: "16px" }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "d-flex gap-1", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx(Skeleton, { width: "70px", height: "32px", borderRadius: "4px" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx(Skeleton, { width: "32px", height: "32px", borderRadius: "4px" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx(Skeleton, { width: "32px", height: "32px", borderRadius: "4px" })
      ] })
    ] }) }),
    /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "card-body p-2", children: /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "d-flex flex-column gap-2", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx(Skeleton, { width: "60%", height: "14px" }),
      /* @__PURE__ */ jsxRuntimeExports.jsx(Skeleton, { width: "100%", height: "20px" }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "border rounded p-2 bg-light", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx(Skeleton, { width: "100%", height: "18px" }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "mt-1 d-flex gap-2", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx(Skeleton, { width: "80px", height: "12px" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx(Skeleton, { width: "80px", height: "12px" })
        ] })
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsx(Skeleton, { width: "100%", height: "36px", borderRadius: "4px" })
    ] }) }),
    /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "card-footer p-1 bg-body-tertiary", children: /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "d-flex justify-content-between align-items-center", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx(Skeleton, { width: "50px", height: "12px" }),
      /* @__PURE__ */ jsxRuntimeExports.jsx(Skeleton, { width: "80px", height: "12px" })
    ] }) })
  ] });
};
const ChannelManager = ({ capture }) => {
  const { data: channels, isLoading } = useChannels(capture.id);
  const createChannel2 = useCreateChannel();
  const startChannel2 = useStartChannel(capture.id);
  const stopChannel2 = useStopChannel(capture.id);
  const toast = useToast();
  const [showNewChannel, setShowNewChannel] = reactExports.useState(false);
  const [newChannelFrequency, setNewChannelFrequency] = reactExports.useState(capture.centerHz);
  const [newChannelMode, setNewChannelMode] = reactExports.useState("wbfm");
  const [newChannelSquelch, setNewChannelSquelch] = reactExports.useState(-60);
  const [newChannelAudioRate, setNewChannelAudioRate] = reactExports.useState(48e3);
  const [copiedUrl, setCopiedUrl] = reactExports.useState(null);
  const [playingChannel, setPlayingChannel] = reactExports.useState(null);
  const audioContextRef = reactExports.useRef(null);
  const streamReaderRef = reactExports.useRef(null);
  const shouldPlayRef = reactExports.useRef(false);
  const nextStartTimeRef = reactExports.useRef(0);
  const stopAudio = () => {
    shouldPlayRef.current = false;
    if (streamReaderRef.current) {
      streamReaderRef.current.cancel();
      streamReaderRef.current = null;
    }
  };
  reactExports.useEffect(() => {
    setNewChannelFrequency(capture.centerHz);
    return () => {
      stopAudio();
    };
  }, [capture.id]);
  const handleCreateChannel = () => {
    const offsetHz = newChannelFrequency - capture.centerHz;
    createChannel2.mutate({
      captureId: capture.id,
      request: {
        mode: newChannelMode,
        offsetHz,
        audioRate: newChannelAudioRate,
        squelchDb: newChannelSquelch
      }
    }, {
      onSuccess: () => {
        setShowNewChannel(false);
        setNewChannelFrequency(capture.centerHz);
        setNewChannelSquelch(-60);
        setNewChannelAudioRate(48e3);
        toast.success("Channel created successfully");
      },
      onError: (error) => {
        toast.error((error == null ? void 0 : error.message) || "Failed to create channel");
      }
    });
  };
  const copyToClipboard = (url) => {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(url).then(() => {
        setCopiedUrl(url);
        setTimeout(() => setCopiedUrl(null), 2e3);
        toast.success("URL copied to clipboard");
      }).catch(() => {
        fallbackCopyToClipboard(url);
      });
    } else {
      fallbackCopyToClipboard(url);
    }
  };
  const fallbackCopyToClipboard = (url) => {
    try {
      const textarea = document.createElement("textarea");
      textarea.value = url;
      textarea.style.position = "fixed";
      textarea.style.left = "-999999px";
      textarea.style.top = "-999999px";
      document.body.appendChild(textarea);
      textarea.focus();
      textarea.select();
      const successful = document.execCommand("copy");
      document.body.removeChild(textarea);
      if (successful) {
        setCopiedUrl(url);
        setTimeout(() => setCopiedUrl(null), 2e3);
        toast.success("URL copied to clipboard");
      } else {
        toast.error("Failed to copy URL");
      }
    } catch (err) {
      console.error("Fallback copy failed:", err);
      toast.error("Failed to copy URL");
    }
  };
  const playPCMAudio = async (channelId) => {
    try {
      if (!audioContextRef.current) {
        audioContextRef.current = new AudioContext({ sampleRate: 48e3 });
      }
      const audioContext = audioContextRef.current;
      shouldPlayRef.current = true;
      nextStartTimeRef.current = audioContext.currentTime;
      const streamUrl = `${window.location.origin}/api/v1/stream/channels/${channelId}.pcm`;
      const response = await fetch(streamUrl);
      if (!response.ok || !response.body) {
        throw new Error("Failed to fetch audio stream");
      }
      const reader = response.body.getReader();
      streamReaderRef.current = reader;
      const bufferSize = 4096;
      let pcmBuffer = [];
      const processChunk = async () => {
        while (shouldPlayRef.current) {
          const { done, value } = await reader.read();
          if (done)
            break;
          const dataView = new DataView(value.buffer, value.byteOffset, value.byteLength);
          const sampleCount = Math.floor(value.length / 2);
          for (let i = 0; i < sampleCount; i++) {
            const sample = dataView.getInt16(i * 2, true) / 32768;
            pcmBuffer.push(sample);
          }
          while (pcmBuffer.length >= bufferSize && shouldPlayRef.current) {
            const chunk = pcmBuffer.splice(0, bufferSize);
            const audioBuffer = audioContext.createBuffer(1, chunk.length, 48e3);
            const channelData = audioBuffer.getChannelData(0);
            for (let i = 0; i < chunk.length; i++) {
              channelData[i] = chunk[i];
            }
            const source = audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(audioContext.destination);
            const startTime = Math.max(nextStartTimeRef.current, audioContext.currentTime);
            source.start(startTime);
            nextStartTimeRef.current = startTime + audioBuffer.duration;
          }
        }
      };
      processChunk().catch((error) => {
        console.error("Audio playback error:", error);
        setPlayingChannel(null);
      });
    } catch (error) {
      console.error("Failed to play audio:", error);
      setPlayingChannel(null);
    }
  };
  const togglePlay = async (channelId) => {
    if (playingChannel === channelId) {
      stopAudio();
      setPlayingChannel(null);
      stopChannel2.mutate(channelId);
      return;
    }
    if (playingChannel) {
      stopAudio();
      setPlayingChannel(null);
    }
    try {
      const channel = channels == null ? void 0 : channels.find((ch2) => ch2.id === channelId);
      if (channel && channel.state !== "running") {
        await startChannel2.mutateAsync(channelId);
      }
      setPlayingChannel(channelId);
      playPCMAudio(channelId);
    } catch (error) {
      console.error("Unable to start channel for playback:", error);
      toast.error((error == null ? void 0 : error.message) || "Failed to start channel");
      setPlayingChannel(null);
    }
  };
  return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "card shadow-sm", children: [
    /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "card-header bg-body-tertiary", children: /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { justify: "between", align: "center", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { align: "center", gap: 2, children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx(Radio, { size: 20 }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("h3", { className: "h6 mb-0", children: "Channels" }),
        channels && channels.length > 0 && /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "badge bg-secondary", children: channels.length })
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsx(
        Button,
        {
          use: "primary",
          size: "sm",
          onClick: () => setShowNewChannel(!showNewChannel),
          disabled: capture.state !== "running",
          children: /* @__PURE__ */ jsxRuntimeExports.jsx(Plus, { size: 16 })
        }
      )
    ] }) }),
    showNewChannel && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "card-body border-bottom", children: /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 3, children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("h6", { className: "mb-0", children: "New Channel" }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 1, children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("label", { className: "form-label small mb-1", children: "Mode" }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs(
          "select",
          {
            className: "form-select form-select-sm",
            value: newChannelMode,
            onChange: (e) => setNewChannelMode(e.target.value),
            children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: "wbfm", children: "WBFM (Wideband FM)" }),
              /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: "nbfm", children: "NBFM (Narrowband FM)" }),
              /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: "am", children: "AM" }),
              /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: "ssb", children: "SSB" }),
              /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: "raw", children: "Raw IQ" }),
              /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: "p25", children: "P25 (Trunked)" }),
              /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: "dmr", children: "DMR (Trunked)" })
            ]
          }
        )
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 1, children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx(
          FrequencySelector,
          {
            label: "Frequency",
            value: newChannelFrequency,
            min: capture.centerHz - capture.sampleRate / 2,
            max: capture.centerHz + capture.sampleRate / 2,
            step: 1e3,
            onChange: setNewChannelFrequency
          }
        ),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("small", { className: "text-muted", children: [
          "Offset: ",
          ((newChannelFrequency - capture.centerHz) / 1e3).toFixed(0),
          " kHz"
        ] })
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsx(
        Slider,
        {
          label: "Squelch",
          value: newChannelSquelch,
          min: -80,
          max: 0,
          step: 1,
          coarseStep: 10,
          unit: "dB",
          formatValue: (val) => `${val.toFixed(0)} dB`,
          onChange: setNewChannelSquelch,
          info: "Signal strength threshold. Lower values (more negative) allow weaker signals."
        }
      ),
      /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 1, children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("label", { className: "form-label small mb-1", children: "Audio Rate" }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs(
          "select",
          {
            className: "form-select form-select-sm",
            value: newChannelAudioRate,
            onChange: (e) => setNewChannelAudioRate(parseInt(e.target.value)),
            children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: 8e3, children: "8 kHz" }),
              /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: 16e3, children: "16 kHz" }),
              /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: 24e3, children: "24 kHz" }),
              /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: 48e3, children: "48 kHz (CD quality)" })
            ]
          }
        )
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { gap: 2, children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx(
          Button,
          {
            use: "success",
            size: "sm",
            onClick: handleCreateChannel,
            disabled: createChannel2.isPending,
            children: "Create"
          }
        ),
        /* @__PURE__ */ jsxRuntimeExports.jsx(
          Button,
          {
            use: "secondary",
            size: "sm",
            onClick: () => setShowNewChannel(false),
            children: "Cancel"
          }
        )
      ] })
    ] }) }),
    /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "card-body", children: [
      isLoading && /* @__PURE__ */ jsxRuntimeExports.jsx(Flex, { justify: "center", className: "py-3", children: /* @__PURE__ */ jsxRuntimeExports.jsx(Spinner, { size: "sm" }) }),
      isLoading && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "row g-3", children: [1, 2].map((i) => /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "col-12 col-xl-6", children: /* @__PURE__ */ jsxRuntimeExports.jsx(SkeletonChannelCard, {}) }, i)) }),
      !isLoading && (!channels || channels.length === 0) && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "text-muted small text-center py-3", children: "No channels. Click + to create one." }),
      !isLoading && channels && channels.length > 0 && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "row g-3", children: channels.map((channel) => /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "col-12 col-xl-6", children: /* @__PURE__ */ jsxRuntimeExports.jsx(
        CompactChannelCard,
        {
          channel,
          capture,
          isPlaying: playingChannel === channel.id,
          onTogglePlay: () => togglePlay(channel.id),
          onCopyUrl: copyToClipboard,
          copiedUrl
        }
      ) }, channel.id)) })
    ] }),
    capture.state !== "running" && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "card-footer bg-body-tertiary", children: /* @__PURE__ */ jsxRuntimeExports.jsx("small", { className: "text-muted", children: "Start the capture to create channels and stream audio." }) })
  ] });
};
const API_BASE = "/api/v1";
async function fetchRecipes() {
  const response = await fetch(`${API_BASE}/recipes`);
  if (!response.ok) {
    throw new Error("Failed to fetch recipes");
  }
  return response.json();
}
function useRecipes() {
  return useQuery({
    queryKey: ["recipes"],
    queryFn: fetchRecipes
  });
}
function CreateCaptureWizard({ onClose, onSuccess }) {
  const { data: recipes, isLoading: recipesLoading } = useRecipes();
  const { data: devices, isLoading: devicesLoading } = useDevices();
  const createCapture2 = useCreateCapture();
  const createChannel2 = useCreateChannel();
  const [step, setStep] = reactExports.useState("select-recipe");
  const [selectedRecipe, setSelectedRecipe] = reactExports.useState(null);
  const [customFrequency, setCustomFrequency] = reactExports.useState(100);
  const [selectedDeviceId, setSelectedDeviceId] = reactExports.useState("");
  const recipesByCategory = (recipes == null ? void 0 : recipes.reduce((acc, recipe) => {
    if (!acc[recipe.category]) {
      acc[recipe.category] = [];
    }
    acc[recipe.category].push(recipe);
    return acc;
  }, {})) || {};
  const handleSelectRecipe = (recipe) => {
    setSelectedRecipe(recipe);
    setCustomFrequency(recipe.centerHz / 1e6);
    setStep("configure");
  };
  const handleCreate = async () => {
    if (!selectedRecipe)
      return;
    const centerHz = selectedRecipe.allowFrequencyInput ? customFrequency * 1e6 : selectedRecipe.centerHz;
    try {
      const newCapture = await createCapture2.mutateAsync({
        deviceId: selectedDeviceId || void 0,
        centerHz,
        sampleRate: selectedRecipe.sampleRate,
        gain: selectedRecipe.gain || void 0,
        bandwidth: selectedRecipe.bandwidth || void 0,
        createDefaultChannel: false
      });
      for (const channelDef of selectedRecipe.channels) {
        await createChannel2.mutateAsync({
          captureId: newCapture.id,
          request: {
            mode: channelDef.mode,
            offsetHz: channelDef.offsetHz,
            squelchDb: channelDef.squelchDb
          }
        });
      }
      onSuccess(newCapture.id);
      onClose();
    } catch (error) {
      console.error("Failed to create capture from recipe:", error);
    }
  };
  const isLoading = recipesLoading || devicesLoading;
  return /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "modal show d-block", style: { backgroundColor: "rgba(0,0,0,0.5)" }, children: /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "modal-dialog modal-lg modal-dialog-centered", children: /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "modal-content", children: [
    /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "modal-header", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { align: "center", gap: 2, children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx(Wand2, { size: 24 }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("h5", { className: "modal-title", children: step === "select-recipe" ? "Choose a Recipe" : `Configure ${selectedRecipe == null ? void 0 : selectedRecipe.name}` })
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("button", { type: "button", className: "btn-close", onClick: onClose })
    ] }),
    /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "modal-body", style: { maxHeight: "70vh", overflowY: "auto" }, children: isLoading ? /* @__PURE__ */ jsxRuntimeExports.jsx(Flex, { justify: "center", align: "center", className: "py-5", children: /* @__PURE__ */ jsxRuntimeExports.jsx(Spinner, {}) }) : step === "select-recipe" ? (
      /* Recipe Selection */
      /* @__PURE__ */ jsxRuntimeExports.jsx(Flex, { direction: "column", gap: 4, children: Object.keys(recipesByCategory).length === 0 ? /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "text-center text-muted py-4", children: "No recipes available. Check your configuration." }) : Object.entries(recipesByCategory).map(([category, categoryRecipes]) => /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("h6", { className: "text-muted mb-3", children: category }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "row g-3", children: categoryRecipes.map((recipe) => /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "col-md-6", children: /* @__PURE__ */ jsxRuntimeExports.jsx(
          "div",
          {
            className: "card h-100 cursor-pointer",
            style: { cursor: "pointer" },
            onClick: () => handleSelectRecipe(recipe),
            children: /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "card-body", children: /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 2, children: [
              /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { align: "center", gap: 2, children: [
                /* @__PURE__ */ jsxRuntimeExports.jsx(Radio, { size: 18, className: "text-primary" }),
                /* @__PURE__ */ jsxRuntimeExports.jsx("h6", { className: "card-title mb-0", children: recipe.name })
              ] }),
              /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "card-text small text-muted mb-0", children: recipe.description }),
              recipe.channels.length > 0 && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "small", children: [
                /* @__PURE__ */ jsxRuntimeExports.jsx("strong", { children: recipe.channels.length }),
                " channel",
                recipe.channels.length !== 1 ? "s" : ""
              ] })
            ] }) })
          }
        ) }, recipe.id)) })
      ] }, category)) })
    ) : (
      /* Configuration */
      /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 4, children: [
        /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "alert alert-info", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("strong", { children: selectedRecipe == null ? void 0 : selectedRecipe.name }),
          /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "mb-0 small", children: selectedRecipe == null ? void 0 : selectedRecipe.description })
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 2, children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("label", { className: "form-label fw-semibold", children: "SDR Device" }),
          /* @__PURE__ */ jsxRuntimeExports.jsxs(
            "select",
            {
              className: "form-select",
              value: selectedDeviceId,
              onChange: (e) => setSelectedDeviceId(e.target.value),
              children: [
                /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: "", children: "Auto-select device" }),
                devices == null ? void 0 : devices.map((device) => /* @__PURE__ */ jsxRuntimeExports.jsx("option", { value: device.id, children: getDeviceDisplayName(device) }, device.id))
              ]
            }
          )
        ] }),
        (selectedRecipe == null ? void 0 : selectedRecipe.allowFrequencyInput) && /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 2, children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("label", { className: "form-label fw-semibold", children: selectedRecipe.frequencyLabel || "Frequency (MHz)" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "input",
            {
              type: "number",
              className: "form-control",
              value: customFrequency,
              onChange: (e) => setCustomFrequency(parseFloat(e.target.value)),
              step: "0.1",
              min: "1",
              max: "6000"
            }
          )
        ] }),
        selectedRecipe && selectedRecipe.channels.length > 0 && /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 2, children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("label", { className: "form-label fw-semibold", children: "Channels to Create" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "list-group", children: selectedRecipe.channels.map((channel, idx) => /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "list-group-item", children: /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { justify: "between", align: "center", children: [
            /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx("strong", { children: channel.name }),
              /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "small text-muted", children: [
                channel.mode.toUpperCase(),
                " • Offset:",
                " ",
                (channel.offsetHz / 1e3).toFixed(0),
                " kHz"
              ] })
            ] }),
            /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "badge bg-secondary", children: channel.mode })
          ] }) }, idx)) })
        ] })
      ] })
    ) }),
    /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "modal-footer", children: [
      step === "configure" && /* @__PURE__ */ jsxRuntimeExports.jsx(Button, { use: "secondary", size: "sm", onClick: () => setStep("select-recipe"), children: "Back" }),
      /* @__PURE__ */ jsxRuntimeExports.jsx(Button, { use: "secondary", size: "sm", onClick: onClose, children: "Cancel" }),
      step === "configure" && /* @__PURE__ */ jsxRuntimeExports.jsx(
        Button,
        {
          use: "success",
          size: "sm",
          onClick: handleCreate,
          disabled: createCapture2.isPending || createChannel2.isPending,
          children: createCapture2.isPending || createChannel2.isPending ? /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
            /* @__PURE__ */ jsxRuntimeExports.jsx(Spinner, { size: "sm" }),
            " Creating..."
          ] }) : "Create Capture"
        }
      )
    ] })
  ] }) }) });
}
class SpectrumWebSocketManager {
  constructor() {
    __publicField(this, "ws", null);
    __publicField(this, "subscribers", /* @__PURE__ */ new Set());
    __publicField(this, "connectionListeners", /* @__PURE__ */ new Set());
    __publicField(this, "captureId", null);
    __publicField(this, "captureState", null);
    __publicField(this, "isIdle", false);
    __publicField(this, "reconnectTimeout", null);
  }
  subscribe(captureId, captureState, isIdle, isPaused, onData, onConnectionChange) {
    if (!isPaused) {
      this.subscribers.add(onData);
    }
    this.connectionListeners.add(onConnectionChange);
    const needsReconnect = this.captureId !== captureId || this.captureState !== captureState || this.isIdle !== isIdle;
    if (needsReconnect) {
      this.captureId = captureId;
      this.captureState = captureState;
      this.isIdle = isIdle;
      this.connect();
    } else if (this.ws && this.ws.readyState === WebSocket.OPEN && !isPaused) {
      onConnectionChange(true);
    } else if (isPaused) {
      if (this.subscribers.size === 0) {
        this.disconnect();
      }
      onConnectionChange(false);
    } else if (!isPaused && this.subscribers.size > 0 && (!this.ws || this.ws.readyState !== WebSocket.OPEN)) {
      this.connect();
    }
    return () => {
      this.subscribers.delete(onData);
      this.connectionListeners.delete(onConnectionChange);
      if (this.subscribers.size === 0) {
        this.disconnect();
      }
    };
  }
  connect() {
    this.disconnect();
    if (this.captureState !== "running" || this.isIdle || !this.captureId) {
      this.notifyConnectionChange(false);
      return;
    }
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/api/v1/stream/captures/${this.captureId}/spectrum`;
    try {
      this.ws = new WebSocket(wsUrl);
      this.ws.onopen = () => {
        console.log("Shared spectrum WebSocket connected");
        this.notifyConnectionChange(true);
      };
      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.notifyData(data);
        } catch (error) {
          console.error("Error parsing spectrum data:", error);
        }
      };
      this.ws.onerror = (error) => {
        console.error("Shared spectrum WebSocket error:", error);
      };
      this.ws.onclose = () => {
        console.log("Shared spectrum WebSocket disconnected");
        this.notifyConnectionChange(false);
        this.ws = null;
        if (this.subscribers.size > 0 && this.captureState === "running" && !this.isIdle) {
          if (this.reconnectTimeout) {
            clearTimeout(this.reconnectTimeout);
          }
          this.reconnectTimeout = setTimeout(() => {
            console.log("Attempting to reconnect spectrum WebSocket...");
            this.connect();
          }, 2e3);
        }
      };
    } catch (error) {
      console.error("Error creating WebSocket:", error);
      this.notifyConnectionChange(false);
    }
  }
  disconnect() {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
    if (this.ws) {
      const ws = this.ws;
      ws.onopen = null;
      ws.onmessage = null;
      ws.onerror = null;
      ws.onclose = null;
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close();
      }
      this.ws = null;
    }
  }
  notifyData(data) {
    this.subscribers.forEach((callback) => callback(data));
  }
  notifyConnectionChange(connected) {
    this.connectionListeners.forEach((callback) => callback(connected));
  }
}
const spectrumManager = new SpectrumWebSocketManager();
function useSpectrumData(capture, isPaused = false) {
  const [spectrumData, setSpectrumData] = reactExports.useState(null);
  const [isConnected, setIsConnected] = reactExports.useState(false);
  const [isIdle, setIsIdle] = reactExports.useState(false);
  const idleTimerRef = reactExports.useRef(null);
  reactExports.useEffect(() => {
    const IDLE_TIMEOUT = 6e4;
    const resetIdleTimer = () => {
      setIsIdle(false);
      if (idleTimerRef.current) {
        clearTimeout(idleTimerRef.current);
      }
      idleTimerRef.current = setTimeout(() => {
        console.log("Spectrum data: UI idle, pausing");
        setIsIdle(true);
      }, IDLE_TIMEOUT);
    };
    const activityEvents = ["mousedown", "mousemove", "keydown", "scroll", "touchstart"];
    activityEvents.forEach((event) => {
      window.addEventListener(event, resetIdleTimer, { passive: true });
    });
    resetIdleTimer();
    return () => {
      activityEvents.forEach((event) => {
        window.removeEventListener(event, resetIdleTimer);
      });
      if (idleTimerRef.current) {
        clearTimeout(idleTimerRef.current);
      }
    };
  }, []);
  reactExports.useEffect(() => {
    if (capture.state !== "running" || isPaused) {
      setSpectrumData(null);
    }
    const unsubscribe = spectrumManager.subscribe(
      capture.id,
      capture.state,
      isIdle,
      isPaused,
      setSpectrumData,
      setIsConnected
    );
    return unsubscribe;
  }, [capture.id, capture.state, isIdle, isPaused]);
  return { spectrumData, isConnected, isIdle };
}
const MIN_HEIGHT$1 = 100;
const MAX_HEIGHT$1 = 600;
const HEIGHT_STEP$1 = 25;
const STORAGE_KEY$1 = "spectrumAnalyzerHeight";
const BAND_PLAN = [
  // AM Broadcast
  { name: "AM Broadcast", startHz: 53e4, endHz: 17e5, color: "rgba(255, 193, 7, 0.2)" },
  // Shortwave Broadcast
  { name: "Shortwave", startHz: 3e6, endHz: 3e7, color: "rgba(111, 66, 193, 0.2)" },
  // Citizens Band (CB)
  { name: "CB Radio", startHz: 26965e3, endHz: 27405e3, color: "rgba(255, 87, 34, 0.2)" },
  // 10m Ham Band
  { name: "10m Ham", startHz: 28e6, endHz: 297e5, color: "rgba(0, 150, 136, 0.2)" },
  // 6m Ham Band
  { name: "6m Ham", startHz: 5e7, endHz: 54e6, color: "rgba(0, 150, 136, 0.2)" },
  // FM Broadcast
  { name: "FM Broadcast", startHz: 88e6, endHz: 108e6, color: "rgba(33, 150, 243, 0.2)" },
  // Aircraft
  { name: "Aircraft", startHz: 108e6, endHz: 137e6, color: "rgba(76, 175, 80, 0.2)" },
  // 2m Ham Band
  { name: "2m Ham", startHz: 144e6, endHz: 148e6, color: "rgba(0, 150, 136, 0.2)" },
  // Marine VHF
  { name: "Marine VHF", startHz: 156e6, endHz: 163e6, color: "rgba(3, 169, 244, 0.2)" },
  // Weather Radio (NOAA)
  { name: "NOAA Weather", startHz: 1624e5, endHz: 16255e4, color: "rgba(255, 152, 0, 0.2)" },
  // Railroad
  { name: "Railroad", startHz: 15981e4, endHz: 161565e3, color: "rgba(121, 85, 72, 0.2)" },
  // Business/Public Safety
  { name: "Business", startHz: 15e7, endHz: 156e6, color: "rgba(158, 158, 158, 0.2)" },
  // 1.25m Ham Band
  { name: "1.25m Ham", startHz: 222e6, endHz: 225e6, color: "rgba(0, 150, 136, 0.2)" },
  // 70cm Ham Band
  { name: "70cm Ham", startHz: 42e7, endHz: 45e7, color: "rgba(0, 150, 136, 0.2)" },
  // FRS/GMRS
  { name: "FRS/GMRS", startHz: 462562500, endHz: 467712500, color: "rgba(233, 30, 99, 0.2)" },
  // Weather Satellites (NOAA APT)
  { name: "Weather Sats", startHz: 137e6, endHz: 138e6, color: "rgba(103, 58, 183, 0.2)" },
  // ISS/Amateur Satellites
  { name: "Sat Downlink", startHz: 1458e5, endHz: 146e6, color: "rgba(63, 81, 181, 0.2)" }
];
function SpectrumAnalyzer({
  capture,
  channels = [],
  height: initialHeight = 200,
  onFrequencyClick
}) {
  const canvasRef = reactExports.useRef(null);
  const containerRef = reactExports.useRef(null);
  const [width, setWidth] = reactExports.useState(800);
  const [isCollapsed, setIsCollapsed] = reactExports.useState(false);
  const [hoverFrequency, setHoverFrequency] = reactExports.useState(null);
  const [hoverPosition, setHoverPosition] = reactExports.useState(null);
  const [height, setHeight] = reactExports.useState(() => {
    const stored = localStorage.getItem(STORAGE_KEY$1);
    return stored ? parseInt(stored, 10) : initialHeight;
  });
  const [peakHoldEnabled, setPeakHoldEnabled] = reactExports.useState(false);
  const [averagingEnabled, setAveragingEnabled] = reactExports.useState(false);
  const [bandPlanEnabled, setBandPlanEnabled] = reactExports.useState(false);
  const peakHoldData = reactExports.useRef([]);
  const avgHistory = reactExports.useRef([]);
  const PEAK_DECAY_RATE = 0.98;
  const AVG_FRAMES = 4;
  reactExports.useEffect(() => {
    localStorage.setItem(STORAGE_KEY$1, height.toString());
  }, [height]);
  const increaseHeight = () => {
    setHeight((prev) => Math.min(MAX_HEIGHT$1, prev + HEIGHT_STEP$1));
  };
  const decreaseHeight = () => {
    setHeight((prev) => Math.max(MIN_HEIGHT$1, prev - HEIGHT_STEP$1));
  };
  const { spectrumData, isConnected, isIdle } = useSpectrumData(capture, isCollapsed);
  reactExports.useEffect(() => {
    const updateWidth = () => {
      if (containerRef.current) {
        const containerWidth = containerRef.current.offsetWidth;
        setWidth(containerWidth - 16);
      }
    };
    updateWidth();
    window.addEventListener("resize", updateWidth);
    return () => window.removeEventListener("resize", updateWidth);
  }, []);
  reactExports.useEffect(() => {
    if (!canvasRef.current) {
      return;
    }
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }
    ctx.fillStyle = "#f8f9fa";
    ctx.fillRect(0, 0, width, height);
    if (!spectrumData) {
      ctx.strokeStyle = "#e9ecef";
      ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i++) {
        const y2 = i / 4 * height;
        ctx.beginPath();
        ctx.moveTo(0, y2);
        ctx.lineTo(width, y2);
        ctx.stroke();
      }
      for (let i = 0; i <= 8; i++) {
        const x2 = i / 8 * width;
        ctx.beginPath();
        ctx.moveTo(x2, 0);
        ctx.lineTo(x2, height);
        ctx.stroke();
      }
      const centerX2 = width / 2;
      ctx.strokeStyle = "#dee2e6";
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(centerX2, 0);
      ctx.lineTo(centerX2, height);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#6c757d";
      ctx.font = "14px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("Capture Stopped", width / 2, height / 2 - 10);
      ctx.font = "12px sans-serif";
      ctx.fillStyle = "#adb5bd";
      ctx.fillText("Click Start to begin capturing spectrum", width / 2, height / 2 + 10);
      return;
    }
    const { power, freqs, centerHz } = spectrumData;
    if (power.length === 0) {
      return;
    }
    const freqMin = centerHz + freqs[0];
    const freqMax = centerHz + freqs[freqs.length - 1];
    const freqMid = centerHz;
    let displayPower = power;
    if (averagingEnabled) {
      avgHistory.current.push([...power]);
      if (avgHistory.current.length > AVG_FRAMES) {
        avgHistory.current.shift();
      }
      if (avgHistory.current.length > 0) {
        displayPower = power.map((_, i) => {
          const sum = avgHistory.current.reduce((acc, frame) => acc + frame[i], 0);
          return sum / avgHistory.current.length;
        });
      }
    } else {
      avgHistory.current = [];
    }
    if (peakHoldEnabled) {
      if (peakHoldData.current.length !== power.length) {
        peakHoldData.current = [...power];
      } else {
        peakHoldData.current = peakHoldData.current.map((peak, i) => {
          const decayed = peak * PEAK_DECAY_RATE;
          return Math.max(decayed, power[i]);
        });
      }
    } else {
      peakHoldData.current = [];
    }
    const minPower = Math.min(...displayPower);
    const maxPower = Math.max(...displayPower);
    const powerRange = maxPower - minPower;
    ctx.strokeStyle = "#0d6efd";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < displayPower.length; i++) {
      const x2 = i / displayPower.length * width;
      const normalized = (displayPower[i] - minPower) / (powerRange || 1);
      const y2 = height - normalized * height;
      if (i === 0) {
        ctx.moveTo(x2, y2);
      } else {
        ctx.lineTo(x2, y2);
      }
    }
    ctx.stroke();
    if (peakHoldEnabled && peakHoldData.current.length > 0) {
      ctx.strokeStyle = "#dc3545";
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.6;
      ctx.beginPath();
      for (let i = 0; i < peakHoldData.current.length; i++) {
        const x2 = i / peakHoldData.current.length * width;
        const normalized = (peakHoldData.current[i] - minPower) / (powerRange || 1);
        const y2 = height - normalized * height;
        if (i === 0) {
          ctx.moveTo(x2, y2);
        } else {
          ctx.lineTo(x2, y2);
        }
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
    ctx.strokeStyle = "#dee2e6";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y2 = i / 4 * height;
      ctx.beginPath();
      ctx.moveTo(0, y2);
      ctx.lineTo(width, y2);
      ctx.stroke();
    }
    for (let i = 0; i <= 8; i++) {
      const x2 = i / 8 * width;
      ctx.beginPath();
      ctx.moveTo(x2, 0);
      ctx.lineTo(x2, height);
      ctx.stroke();
    }
    const centerX = width / 2;
    ctx.strokeStyle = "#dc3545";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(centerX, 0);
    ctx.lineTo(centerX, height);
    ctx.stroke();
    if (bandPlanEnabled) {
      const freqSpan = freqMax - freqMin;
      BAND_PLAN.forEach((band) => {
        if (band.endHz >= freqMin && band.startHz <= freqMax) {
          const bandStart = Math.max(band.startHz, freqMin);
          const bandEnd = Math.min(band.endHz, freqMax);
          const x1 = (bandStart - freqMin) / freqSpan * width;
          const x2 = (bandEnd - freqMin) / freqSpan * width;
          const bandWidth = x2 - x1;
          if (bandWidth >= 5) {
            ctx.fillStyle = band.color;
            ctx.fillRect(x1, 0, bandWidth, height);
            ctx.strokeStyle = band.color.replace("0.2)", "0.5)");
            ctx.lineWidth = 1;
            ctx.strokeRect(x1, 0, bandWidth, height);
            if (bandWidth >= 40) {
              ctx.font = "bold 10px sans-serif";
              ctx.fillStyle = band.color.replace("0.2)", "0.9)").replace("rgba", "rgb").replace(/, 0\.9\)/, ")");
              ctx.textAlign = "center";
              ctx.textBaseline = "top";
              const labelText = band.name;
              const labelWidth = ctx.measureText(labelText).width;
              const labelX = x1 + bandWidth / 2;
              ctx.fillStyle = "rgba(255, 255, 255, 0.85)";
              ctx.fillRect(labelX - labelWidth / 2 - 3, 5, labelWidth + 6, 14);
              ctx.fillStyle = "#000";
              ctx.fillText(labelText, labelX, 8);
            }
          }
        }
      });
    }
    ctx.font = "10px monospace";
    const freqMinText = `${(freqMin / 1e6).toFixed(3)} MHz`;
    const freqMidText = `${(freqMid / 1e6).toFixed(3)} MHz`;
    const freqMaxText = `${(freqMax / 1e6).toFixed(3)} MHz`;
    ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
    const freqMinWidth = ctx.measureText(freqMinText).width;
    const freqMidWidth = ctx.measureText(freqMidText).width;
    const freqMaxWidth = ctx.measureText(freqMaxText).width;
    ctx.fillRect(3, height - 16, freqMinWidth + 4, 13);
    ctx.fillRect(width / 2 - 32, height - 16, freqMidWidth + 4, 13);
    ctx.fillRect(width - 72, height - 16, freqMaxWidth + 4, 13);
    ctx.fillStyle = "#6c757d";
    ctx.fillText(freqMinText, 5, height - 5);
    ctx.fillText(freqMidText, width / 2 - 30, height - 5);
    ctx.fillText(freqMaxText, width - 70, height - 5);
    const maxPowerText = `${maxPower.toFixed(1)} dB`;
    const minPowerText = `${minPower.toFixed(1)} dB`;
    const maxPowerWidth = ctx.measureText(maxPowerText).width;
    const minPowerWidth = ctx.measureText(minPowerText).width;
    ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
    ctx.fillRect(3, 1, maxPowerWidth + 4, 13);
    ctx.fillRect(3, height - 26, minPowerWidth + 4, 13);
    ctx.fillStyle = "#6c757d";
    ctx.fillText(maxPowerText, 5, 12);
    ctx.fillText(minPowerText, 5, height - 15);
    const formatRate = (rate) => {
      if (rate >= 1e9)
        return `${(rate / 1e9).toFixed(2)} GS/s`;
      if (rate >= 1e6)
        return `${(rate / 1e6).toFixed(2)} MS/s`;
      if (rate >= 1e3)
        return `${(rate / 1e3).toFixed(2)} kS/s`;
      return `${rate.toFixed(0)} S/s`;
    };
    const formatBandwidth = (bw) => {
      if (bw >= 1e9)
        return `${(bw / 1e9).toFixed(2)} GHz`;
      if (bw >= 1e6)
        return `${(bw / 1e6).toFixed(2)} MHz`;
      if (bw >= 1e3)
        return `${(bw / 1e3).toFixed(2)} kHz`;
      return `${bw.toFixed(0)} Hz`;
    };
    const sampleRateY = height * 0.1;
    const overlayHeight = 15;
    ctx.fillStyle = "rgba(0, 123, 255, 0.15)";
    ctx.fillRect(0, sampleRateY, width, overlayHeight);
    ctx.strokeStyle = "rgba(0, 123, 255, 0.5)";
    ctx.lineWidth = 1;
    ctx.strokeRect(0, sampleRateY, width, overlayHeight);
    ctx.font = "bold 9px monospace";
    const sampleRateText = `Sample Rate: ${formatRate(capture.sampleRate)}`;
    const sampleRateTextWidth = ctx.measureText(sampleRateText).width;
    const sampleRateTextX = (width - sampleRateTextWidth) / 2;
    ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
    ctx.fillRect(sampleRateTextX - 2, sampleRateY + 2, sampleRateTextWidth + 4, 11);
    ctx.fillStyle = "#007bff";
    ctx.fillText(sampleRateText, sampleRateTextX, sampleRateY + 10);
    if (capture.bandwidth !== null && capture.bandwidth !== void 0) {
      const bandwidthY = height * 0.18;
      const freqSpan = freqMax - freqMin;
      const bandwidthWidth = capture.bandwidth / freqSpan * width;
      const bandwidthX = (width - bandwidthWidth) / 2;
      ctx.fillStyle = "rgba(40, 167, 69, 0.15)";
      ctx.fillRect(bandwidthX, bandwidthY, bandwidthWidth, overlayHeight);
      ctx.strokeStyle = "rgba(40, 167, 69, 0.5)";
      ctx.lineWidth = 1;
      ctx.strokeRect(bandwidthX, bandwidthY, bandwidthWidth, overlayHeight);
      ctx.font = "bold 9px monospace";
      const bandwidthText = `Bandwidth: ${formatBandwidth(capture.bandwidth)}`;
      const bandwidthTextWidth = ctx.measureText(bandwidthText).width;
      const bandwidthTextX = (width - bandwidthTextWidth) / 2;
      ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
      ctx.fillRect(bandwidthTextX - 2, bandwidthY + 2, bandwidthTextWidth + 4, 11);
      ctx.fillStyle = "#28a745";
      ctx.fillText(bandwidthText, bandwidthTextX, bandwidthY + 10);
    }
    if (channels && channels.length > 0) {
      const freqSpan = freqMax - freqMin;
      channels.forEach((channel, idx) => {
        const channelFreq = centerHz + channel.offsetHz;
        const freqOffset = channelFreq - freqMin;
        const x2 = freqOffset / freqSpan * width;
        if (x2 >= 0 && x2 <= width) {
          ctx.strokeStyle = channel.state === "running" ? "#198754" : "#adb5bd";
          ctx.lineWidth = 2;
          ctx.setLineDash([5, 3]);
          ctx.beginPath();
          ctx.moveTo(x2, 0);
          ctx.lineTo(x2, height);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.font = "bold 10px monospace";
          const label = `CH${idx + 1}`;
          const labelWidth = ctx.measureText(label).width;
          ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
          ctx.fillRect(x2 - labelWidth / 2 - 2, 0, labelWidth + 4, 12);
          ctx.fillStyle = channel.state === "running" ? "#198754" : "#6c757d";
          ctx.fillText(label, x2 - labelWidth / 2, 10);
        }
      });
    }
  }, [spectrumData, width, height, channels, capture.bandwidth, capture.sampleRate, peakHoldEnabled, averagingEnabled, bandPlanEnabled]);
  const getBadgeStatus = () => {
    if (isIdle && capture.state === "running") {
      return { text: "PAUSED (IDLE)", className: "bg-warning" };
    } else if (isConnected) {
      return { text: "LIVE", className: "bg-success" };
    } else {
      return { text: "OFFLINE", className: "bg-secondary" };
    }
  };
  const badgeStatus = getBadgeStatus();
  const handleCanvasClick = (event) => {
    if (!onFrequencyClick || !spectrumData)
      return;
    const canvas = canvasRef.current;
    if (!canvas)
      return;
    const rect = canvas.getBoundingClientRect();
    const x2 = event.clientX - rect.left;
    const { freqs, centerHz } = spectrumData;
    const freqMin = centerHz + freqs[0];
    const freqMax = centerHz + freqs[freqs.length - 1];
    const freqSpan = freqMax - freqMin;
    const clickedFrequency = freqMin + x2 / width * freqSpan;
    onFrequencyClick(Math.round(clickedFrequency));
  };
  const handleCanvasMouseMove = (event) => {
    if (!spectrumData)
      return;
    const canvas = canvasRef.current;
    if (!canvas)
      return;
    const rect = canvas.getBoundingClientRect();
    const x2 = event.clientX - rect.left;
    const { freqs, centerHz } = spectrumData;
    const freqMin = centerHz + freqs[0];
    const freqMax = centerHz + freqs[freqs.length - 1];
    const freqSpan = freqMax - freqMin;
    const frequency = freqMin + x2 / width * freqSpan;
    setHoverFrequency(frequency);
    setHoverPosition({ x: event.clientX, y: event.clientY });
  };
  const handleCanvasMouseLeave = () => {
    setHoverFrequency(null);
    setHoverPosition(null);
  };
  return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "card shadow-sm", children: [
    /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "card-header bg-body-tertiary py-1 px-2", children: /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "d-flex justify-content-between align-items-center", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("small", { className: "fw-semibold mb-0", children: "Spectrum Analyzer" }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "d-flex align-items-center gap-2", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx(
          "span",
          {
            className: `badge ${badgeStatus.className} text-white`,
            style: { fontSize: "8px", padding: "2px 6px" },
            children: badgeStatus.text
          }
        ),
        !isCollapsed && /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
          /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "d-flex align-items-center gap-1", style: { fontSize: "10px" }, children: [
            /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "d-flex align-items-center gap-1 mb-0", style: { cursor: "pointer" }, children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx(
                "input",
                {
                  type: "checkbox",
                  checked: peakHoldEnabled,
                  onChange: (e) => setPeakHoldEnabled(e.target.checked),
                  style: { width: "12px", height: "12px" }
                }
              ),
              /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "text-muted", children: "Peak" })
            ] }),
            /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "d-flex align-items-center gap-1 mb-0", style: { cursor: "pointer" }, children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx(
                "input",
                {
                  type: "checkbox",
                  checked: averagingEnabled,
                  onChange: (e) => setAveragingEnabled(e.target.checked),
                  style: { width: "12px", height: "12px" }
                }
              ),
              /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "text-muted", children: "Avg" })
            ] }),
            /* @__PURE__ */ jsxRuntimeExports.jsxs("label", { className: "d-flex align-items-center gap-1 mb-0", style: { cursor: "pointer" }, children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx(
                "input",
                {
                  type: "checkbox",
                  checked: bandPlanEnabled,
                  onChange: (e) => setBandPlanEnabled(e.target.checked),
                  style: { width: "12px", height: "12px" }
                }
              ),
              /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "text-muted", children: "Bands" })
            ] })
          ] }),
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "button",
            {
              className: "btn btn-sm btn-outline-secondary p-0",
              style: { width: "20px", height: "20px", lineHeight: 1 },
              onClick: decreaseHeight,
              disabled: height <= MIN_HEIGHT$1,
              title: "Decrease height",
              children: /* @__PURE__ */ jsxRuntimeExports.jsx(Minus, { size: 12 })
            }
          ),
          /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "small text-muted", style: { fontSize: "10px", minWidth: "35px", textAlign: "center" }, children: [
            height,
            "px"
          ] }),
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "button",
            {
              className: "btn btn-sm btn-outline-secondary p-0",
              style: { width: "20px", height: "20px", lineHeight: 1 },
              onClick: increaseHeight,
              disabled: height >= MAX_HEIGHT$1,
              title: "Increase height",
              children: /* @__PURE__ */ jsxRuntimeExports.jsx(Plus, { size: 12 })
            }
          )
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsx(
          "button",
          {
            className: "btn btn-sm btn-outline-secondary p-0",
            style: { width: "20px", height: "20px", lineHeight: 1 },
            onClick: () => setIsCollapsed(!isCollapsed),
            title: isCollapsed ? "Expand" : "Collapse",
            children: isCollapsed ? /* @__PURE__ */ jsxRuntimeExports.jsx(ChevronDown, { size: 14 }) : /* @__PURE__ */ jsxRuntimeExports.jsx(ChevronUp, { size: 14 })
          }
        )
      ] })
    ] }) }),
    !isCollapsed && /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "card-body", ref: containerRef, style: { padding: "0.5rem", position: "relative" }, children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx(
        "canvas",
        {
          ref: canvasRef,
          width,
          height,
          onClick: handleCanvasClick,
          onMouseMove: handleCanvasMouseMove,
          onMouseLeave: handleCanvasMouseLeave,
          style: {
            border: "1px solid #dee2e6",
            borderRadius: "4px",
            display: "block",
            width: "100%",
            cursor: onFrequencyClick ? "crosshair" : "default"
          }
        }
      ),
      hoverFrequency !== null && hoverPosition !== null && /* @__PURE__ */ jsxRuntimeExports.jsxs(
        "div",
        {
          style: {
            position: "fixed",
            left: `${hoverPosition.x + 10}px`,
            top: `${hoverPosition.y - 30}px`,
            backgroundColor: "rgba(0, 0, 0, 0.85)",
            color: "white",
            padding: "4px 8px",
            borderRadius: "4px",
            fontSize: "12px",
            fontWeight: 600,
            fontFamily: "monospace",
            pointerEvents: "none",
            zIndex: 1e3,
            whiteSpace: "nowrap",
            boxShadow: "0 2px 8px rgba(0,0,0,0.3)"
          },
          children: [
            (hoverFrequency / 1e6).toFixed(4),
            " MHz",
            onFrequencyClick && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { style: { fontSize: "10px", opacity: 0.8, marginTop: "2px" }, children: "Click to tune" })
          ]
        }
      )
    ] })
  ] });
}
const MIN_HEIGHT = 100;
const MAX_HEIGHT = 600;
const HEIGHT_STEP = 25;
const STORAGE_KEY = "waterfallDisplayHeight";
const COLOR_SCHEMES = {
  heat: [
    { pos: 0, r: 0, g: 0, b: 128 },
    // Dark blue (low power)
    { pos: 0.25, r: 0, g: 0, b: 255 },
    // Blue
    { pos: 0.5, r: 0, g: 255, b: 255 },
    // Cyan
    { pos: 0.75, r: 255, g: 255, b: 0 },
    // Yellow
    { pos: 1, r: 255, g: 0, b: 0 }
    // Red (high power)
  ],
  grayscale: [
    { pos: 0, r: 0, g: 0, b: 0 },
    // Black (low power)
    { pos: 1, r: 255, g: 255, b: 255 }
    // White (high power)
  ],
  viridis: [
    { pos: 0, r: 68, g: 1, b: 84 },
    // Dark purple
    { pos: 0.25, r: 59, g: 82, b: 139 },
    // Blue
    { pos: 0.5, r: 33, g: 145, b: 140 },
    // Teal
    { pos: 0.75, r: 94, g: 201, b: 98 },
    // Green
    { pos: 1, r: 253, g: 231, b: 37 }
    // Yellow
  ]
};
function WaterfallDisplay({
  capture,
  channels = [],
  height: initialHeight = 300,
  timeSpanSeconds = 30,
  colorScheme = "heat",
  intensity = 1
}) {
  const canvasRef = reactExports.useRef(null);
  const containerRef = reactExports.useRef(null);
  const [width, setWidth] = reactExports.useState(800);
  const [isCollapsed, setIsCollapsed] = reactExports.useState(false);
  const [height, setHeight] = reactExports.useState(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    return stored ? parseInt(stored, 10) : initialHeight;
  });
  reactExports.useEffect(() => {
    localStorage.setItem(STORAGE_KEY, height.toString());
  }, [height]);
  const increaseHeight = () => {
    setHeight((prev) => Math.min(MAX_HEIGHT, prev + HEIGHT_STEP));
  };
  const decreaseHeight = () => {
    setHeight((prev) => Math.max(MIN_HEIGHT, prev - HEIGHT_STEP));
  };
  const { spectrumData, isConnected, isIdle } = useSpectrumData(capture, isCollapsed);
  const historyRef = reactExports.useRef([]);
  const maxHistoryLines = height;
  const [spectrumInfo, setSpectrumInfo] = reactExports.useState(null);
  const renderRequestRef = reactExports.useRef(null);
  const needsRenderRef = reactExports.useRef(false);
  reactExports.useEffect(() => {
    const updateWidth = () => {
      if (containerRef.current) {
        const containerWidth = containerRef.current.offsetWidth;
        setWidth(containerWidth - 16);
      }
    };
    updateWidth();
    window.addEventListener("resize", updateWidth);
    return () => window.removeEventListener("resize", updateWidth);
  }, []);
  reactExports.useEffect(() => {
    if (!spectrumData) {
      if (capture.state !== "running") {
        historyRef.current = [];
        setSpectrumInfo(null);
        needsRenderRef.current = true;
      }
      return;
    }
    historyRef.current.push(spectrumData.power);
    if (historyRef.current.length > maxHistoryLines) {
      historyRef.current.shift();
    }
    const minPower = Math.min(...spectrumData.power);
    const maxPower = Math.max(...spectrumData.power);
    setSpectrumInfo({
      centerHz: spectrumData.centerHz,
      freqs: spectrumData.freqs,
      minPower,
      maxPower
    });
    needsRenderRef.current = true;
  }, [spectrumData, capture.state, maxHistoryLines]);
  const getColor = reactExports.useCallback((normalizedValue) => {
    const adjustedValue = Math.pow(normalizedValue, 1 / intensity);
    const clampedValue = Math.max(0, Math.min(1, adjustedValue));
    const gradient = COLOR_SCHEMES[colorScheme];
    let lowerStop = gradient[0];
    let upperStop = gradient[gradient.length - 1];
    for (let i = 0; i < gradient.length - 1; i++) {
      if (clampedValue >= gradient[i].pos && clampedValue <= gradient[i + 1].pos) {
        lowerStop = gradient[i];
        upperStop = gradient[i + 1];
        break;
      }
    }
    const range = upperStop.pos - lowerStop.pos;
    const t2 = range === 0 ? 0 : (clampedValue - lowerStop.pos) / range;
    const r2 = Math.round(lowerStop.r + t2 * (upperStop.r - lowerStop.r));
    const g = Math.round(lowerStop.g + t2 * (upperStop.g - lowerStop.g));
    const b = Math.round(lowerStop.b + t2 * (upperStop.b - lowerStop.b));
    return [r2, g, b];
  }, [colorScheme, intensity]);
  reactExports.useEffect(() => {
    const render = () => {
      if (!needsRenderRef.current || !canvasRef.current) {
        renderRequestRef.current = requestAnimationFrame(render);
        return;
      }
      needsRenderRef.current = false;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        renderRequestRef.current = requestAnimationFrame(render);
        return;
      }
      ctx.fillStyle = "#1a1a1a";
      ctx.fillRect(0, 0, width, height);
      const history = historyRef.current;
      if (history.length === 0) {
        ctx.fillStyle = "#6c757d";
        ctx.font = "14px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(
          capture.state === "running" ? "Waiting for data..." : "Capture Stopped",
          width / 2,
          height / 2 - 10
        );
        ctx.font = "12px sans-serif";
        ctx.fillStyle = "#adb5bd";
        ctx.fillText(
          capture.state === "running" ? "Waterfall will appear shortly" : "Click Start to begin capturing",
          width / 2,
          height / 2 + 10
        );
        renderRequestRef.current = requestAnimationFrame(render);
        return;
      }
      let globalMin = Infinity;
      let globalMax = -Infinity;
      history.forEach((powerArray) => {
        const min = Math.min(...powerArray);
        const max = Math.max(...powerArray);
        if (min < globalMin)
          globalMin = min;
        if (max > globalMax)
          globalMax = max;
      });
      const powerRange = globalMax - globalMin || 1;
      const imageData = ctx.createImageData(width, height);
      const data = imageData.data;
      for (let lineIdx = 0; lineIdx < history.length; lineIdx++) {
        const powerArray = history[lineIdx];
        const y2 = height - 1 - lineIdx;
        for (let x2 = 0; x2 < width; x2++) {
          const binIdx = Math.floor(x2 / width * powerArray.length);
          const power = powerArray[binIdx];
          const normalized = (power - globalMin) / powerRange;
          const [r2, g, b] = getColor(normalized);
          const pixelIdx = (y2 * width + x2) * 4;
          data[pixelIdx] = r2;
          data[pixelIdx + 1] = g;
          data[pixelIdx + 2] = b;
          data[pixelIdx + 3] = 255;
        }
      }
      ctx.putImageData(imageData, 0, 0);
      const formatRate = (rate) => {
        if (rate >= 1e9)
          return `${(rate / 1e9).toFixed(2)} GS/s`;
        if (rate >= 1e6)
          return `${(rate / 1e6).toFixed(2)} MS/s`;
        if (rate >= 1e3)
          return `${(rate / 1e3).toFixed(2)} kS/s`;
        return `${rate.toFixed(0)} S/s`;
      };
      const formatBandwidth = (bw) => {
        if (bw >= 1e9)
          return `${(bw / 1e9).toFixed(2)} GHz`;
        if (bw >= 1e6)
          return `${(bw / 1e6).toFixed(2)} MHz`;
        if (bw >= 1e3)
          return `${(bw / 1e3).toFixed(2)} kHz`;
        return `${bw.toFixed(0)} Hz`;
      };
      if (spectrumInfo && spectrumData) {
        const sampleRateY = height * 0.05;
        const overlayHeight = 15;
        ctx.fillStyle = "rgba(0, 123, 255, 0.15)";
        ctx.fillRect(0, sampleRateY, width, overlayHeight);
        ctx.strokeStyle = "rgba(0, 123, 255, 0.5)";
        ctx.lineWidth = 1;
        ctx.strokeRect(0, sampleRateY, width, overlayHeight);
        ctx.font = "bold 9px monospace";
        const sampleRateText = `Sample Rate: ${formatRate(spectrumData.sampleRate)}`;
        const sampleRateTextWidth = ctx.measureText(sampleRateText).width;
        const sampleRateTextX = (width - sampleRateTextWidth) / 2;
        ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
        ctx.fillRect(sampleRateTextX - 2, sampleRateY + 2, sampleRateTextWidth + 4, 11);
        ctx.fillStyle = "#007bff";
        ctx.fillText(sampleRateText, sampleRateTextX, sampleRateY + 10);
        if (capture.bandwidth !== null && capture.bandwidth !== void 0) {
          const bandwidthY = height * 0.13;
          const { centerHz, freqs } = spectrumInfo;
          const freqMin = centerHz + freqs[0];
          const freqMax = centerHz + freqs[freqs.length - 1];
          const freqSpan = freqMax - freqMin;
          const bandwidthWidth = capture.bandwidth / freqSpan * width;
          const bandwidthX = (width - bandwidthWidth) / 2;
          ctx.fillStyle = "rgba(40, 167, 69, 0.15)";
          ctx.fillRect(bandwidthX, bandwidthY, bandwidthWidth, overlayHeight);
          ctx.strokeStyle = "rgba(40, 167, 69, 0.5)";
          ctx.lineWidth = 1;
          ctx.strokeRect(bandwidthX, bandwidthY, bandwidthWidth, overlayHeight);
          ctx.font = "bold 9px monospace";
          const bandwidthText = `Bandwidth: ${formatBandwidth(capture.bandwidth)}`;
          const bandwidthTextWidth = ctx.measureText(bandwidthText).width;
          const bandwidthTextX = (width - bandwidthTextWidth) / 2;
          ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
          ctx.fillRect(bandwidthTextX - 2, bandwidthY + 2, bandwidthTextWidth + 4, 11);
          ctx.fillStyle = "#28a745";
          ctx.fillText(bandwidthText, bandwidthTextX, bandwidthY + 10);
        }
      }
      if (spectrumInfo && channels && channels.length > 0) {
        const { centerHz, freqs } = spectrumInfo;
        const freqMin = centerHz + freqs[0];
        const freqMax = centerHz + freqs[freqs.length - 1];
        const freqSpan = freqMax - freqMin;
        channels.forEach((channel, idx) => {
          const channelFreq = centerHz + channel.offsetHz;
          const freqOffset = channelFreq - freqMin;
          const x2 = freqOffset / freqSpan * width;
          if (x2 >= 0 && x2 <= width) {
            ctx.strokeStyle = channel.state === "running" ? "#00ff00" : "#808080";
            ctx.lineWidth = 1;
            ctx.setLineDash([3, 2]);
            ctx.beginPath();
            ctx.moveTo(x2, 0);
            ctx.lineTo(x2, height);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.font = "bold 9px monospace";
            const label = `CH${idx + 1}`;
            const labelWidth = ctx.measureText(label).width;
            ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
            ctx.fillRect(x2 - labelWidth / 2 - 2, 2, labelWidth + 4, 12);
            ctx.fillStyle = channel.state === "running" ? "#198754" : "#6c757d";
            ctx.fillText(label, x2 - labelWidth / 2, 11);
          }
        });
      }
      const centerX = width / 2;
      ctx.strokeStyle = "rgba(255, 0, 0, 0.5)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(centerX, 0);
      ctx.lineTo(centerX, height);
      ctx.stroke();
      if (spectrumInfo) {
        const { centerHz, freqs } = spectrumInfo;
        const freqMin = centerHz + freqs[0];
        const freqMax = centerHz + freqs[freqs.length - 1];
        ctx.font = "10px monospace";
        const freqMinText = `${(freqMin / 1e6).toFixed(3)} MHz`;
        const freqMidText = `${(centerHz / 1e6).toFixed(3)} MHz`;
        const freqMaxText = `${(freqMax / 1e6).toFixed(3)} MHz`;
        const freqMinWidth = ctx.measureText(freqMinText).width;
        const freqMidWidth = ctx.measureText(freqMidText).width;
        const freqMaxWidth = ctx.measureText(freqMaxText).width;
        ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
        ctx.fillRect(3, height - 16, freqMinWidth + 4, 13);
        ctx.fillRect(width / 2 - 32, height - 16, freqMidWidth + 4, 13);
        ctx.fillRect(width - 72, height - 16, freqMaxWidth + 4, 13);
        ctx.fillStyle = "#000000";
        ctx.fillText(freqMinText, 5, height - 5);
        ctx.fillText(freqMidText, width / 2 - 30, height - 5);
        ctx.fillText(freqMaxText, width - 70, height - 5);
      }
      renderRequestRef.current = requestAnimationFrame(render);
    };
    renderRequestRef.current = requestAnimationFrame(render);
    return () => {
      if (renderRequestRef.current !== null) {
        cancelAnimationFrame(renderRequestRef.current);
      }
    };
  }, [width, height, channels, spectrumInfo, colorScheme, intensity, capture.state, capture.bandwidth, getColor]);
  const getBadgeStatus = () => {
    if (isIdle && capture.state === "running") {
      return { text: "PAUSED (IDLE)", className: "bg-warning" };
    } else if (isConnected) {
      return { text: "LIVE", className: "bg-success" };
    } else {
      return { text: "OFFLINE", className: "bg-secondary" };
    }
  };
  const badgeStatus = getBadgeStatus();
  return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "card shadow-sm", children: [
    /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "card-header bg-body-tertiary py-1 px-2", children: /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "d-flex justify-content-between align-items-center", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsxs("small", { className: "fw-semibold mb-0", children: [
        "Waterfall Display (",
        timeSpanSeconds,
        "s)"
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "d-flex align-items-center gap-2", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx(
          "span",
          {
            className: `badge ${badgeStatus.className} text-white`,
            style: { fontSize: "8px", padding: "2px 6px" },
            children: badgeStatus.text
          }
        ),
        !isCollapsed && /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "button",
            {
              className: "btn btn-sm btn-outline-secondary p-0",
              style: { width: "20px", height: "20px", lineHeight: 1 },
              onClick: decreaseHeight,
              disabled: height <= MIN_HEIGHT,
              title: "Decrease height",
              children: /* @__PURE__ */ jsxRuntimeExports.jsx(Minus, { size: 12 })
            }
          ),
          /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "small text-muted", style: { fontSize: "10px", minWidth: "35px", textAlign: "center" }, children: [
            height,
            "px"
          ] }),
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "button",
            {
              className: "btn btn-sm btn-outline-secondary p-0",
              style: { width: "20px", height: "20px", lineHeight: 1 },
              onClick: increaseHeight,
              disabled: height >= MAX_HEIGHT,
              title: "Increase height",
              children: /* @__PURE__ */ jsxRuntimeExports.jsx(Plus, { size: 12 })
            }
          )
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsx(
          "button",
          {
            className: "btn btn-sm btn-outline-secondary p-0",
            style: { width: "20px", height: "20px", lineHeight: 1 },
            onClick: () => setIsCollapsed(!isCollapsed),
            title: isCollapsed ? "Expand" : "Collapse",
            children: isCollapsed ? /* @__PURE__ */ jsxRuntimeExports.jsx(ChevronDown, { size: 14 }) : /* @__PURE__ */ jsxRuntimeExports.jsx(ChevronUp, { size: 14 })
          }
        )
      ] })
    ] }) }),
    !isCollapsed && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "card-body", ref: containerRef, style: { padding: "0.5rem" }, children: /* @__PURE__ */ jsxRuntimeExports.jsx(
      "canvas",
      {
        ref: canvasRef,
        width,
        height,
        style: {
          border: "1px solid #dee2e6",
          borderRadius: "4px",
          display: "block",
          width: "100%",
          imageRendering: "pixelated"
          // Crisp pixels for waterfall
        }
      }
    ) })
  ] });
}
async function updateDeviceNickname(deviceId, nickname) {
  const response = await fetch(`/api/v1/devices/${deviceId}/name`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ nickname })
  });
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Failed to update device nickname");
  }
  return response.json();
}
function useUpdateDeviceNickname() {
  const queryClient2 = useQueryClient();
  return useMutation({
    mutationFn: ({ deviceId, nickname }) => updateDeviceNickname(deviceId, nickname),
    onSuccess: () => {
      queryClient2.invalidateQueries({ queryKey: ["devices"] });
    }
  });
}
const DeviceSettingsModal = ({ onClose }) => {
  const { data: devices, isLoading } = useDevices();
  const updateNickname = useUpdateDeviceNickname();
  const [nicknames, setNicknames] = reactExports.useState({});
  const [hasChanges, setHasChanges] = reactExports.useState(false);
  reactExports.useEffect(() => {
    if (devices) {
      const initialNicknames = {};
      devices.forEach((device) => {
        initialNicknames[device.id] = device.nickname || "";
      });
      setNicknames(initialNicknames);
    }
  }, [devices]);
  const handleNicknameChange = (deviceId, value) => {
    setNicknames((prev) => ({ ...prev, [deviceId]: value }));
    setHasChanges(true);
  };
  const handleSave = async () => {
    if (!devices)
      return;
    const updates = devices.filter((device) => {
      var _a2;
      const newNickname = ((_a2 = nicknames[device.id]) == null ? void 0 : _a2.trim()) || null;
      const oldNickname = device.nickname || null;
      return newNickname !== oldNickname;
    }).map((device) => {
      var _a2;
      return {
        deviceId: device.id,
        nickname: ((_a2 = nicknames[device.id]) == null ? void 0 : _a2.trim()) || null
      };
    });
    for (const update of updates) {
      await updateNickname.mutateAsync(update);
    }
    setHasChanges(false);
    onClose();
  };
  const getDeviceShorthand = (device) => {
    const driver = device.driver.toUpperCase();
    const label = device.label || "";
    const serialMatch = label.match(/SN:?\s*([A-Za-z0-9]+)/i);
    if (serialMatch) {
      return `${driver} (${serialMatch[1]})`;
    }
    const shortLabel = label.length > 40 ? label.substring(0, 40) + "..." : label;
    return `${driver} - ${shortLabel}`;
  };
  return /* @__PURE__ */ jsxRuntimeExports.jsx(
    "div",
    {
      className: "modal d-block",
      style: { backgroundColor: "rgba(0,0,0,0.5)" },
      onClick: onClose,
      children: /* @__PURE__ */ jsxRuntimeExports.jsx(
        "div",
        {
          className: "modal-dialog modal-dialog-centered modal-lg",
          onClick: (e) => e.stopPropagation(),
          children: /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "modal-content", children: [
            /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "modal-header", children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx("h5", { className: "modal-title", children: "Device Settings" }),
              /* @__PURE__ */ jsxRuntimeExports.jsx(
                "button",
                {
                  type: "button",
                  className: "btn-close",
                  onClick: onClose,
                  "aria-label": "Close"
                }
              )
            ] }),
            /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "modal-body", children: isLoading ? /* @__PURE__ */ jsxRuntimeExports.jsx(Flex, { justify: "center", className: "py-4", children: /* @__PURE__ */ jsxRuntimeExports.jsx(Spinner, {}) }) : devices && devices.length > 0 ? /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "text-muted small mb-3", children: "Customize device nicknames to make them easier to identify. Leave blank to use auto-detected names." }),
              /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "table-responsive", children: /* @__PURE__ */ jsxRuntimeExports.jsxs("table", { className: "table table-hover", children: [
                /* @__PURE__ */ jsxRuntimeExports.jsx("thead", { children: /* @__PURE__ */ jsxRuntimeExports.jsxs("tr", { children: [
                  /* @__PURE__ */ jsxRuntimeExports.jsx("th", { style: { width: "35%" }, children: "Device" }),
                  /* @__PURE__ */ jsxRuntimeExports.jsx("th", { style: { width: "40%" }, children: "Auto-Detected" }),
                  /* @__PURE__ */ jsxRuntimeExports.jsx("th", { style: { width: "25%" }, children: "Nickname" })
                ] }) }),
                /* @__PURE__ */ jsxRuntimeExports.jsx("tbody", { children: devices.map((device) => /* @__PURE__ */ jsxRuntimeExports.jsxs("tr", { children: [
                  /* @__PURE__ */ jsxRuntimeExports.jsxs("td", { children: [
                    /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "fw-semibold", children: device.driver.toUpperCase() }),
                    /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "small text-muted", children: [
                      "ID: ",
                      device.id
                    ] })
                  ] }),
                  /* @__PURE__ */ jsxRuntimeExports.jsx("td", { children: /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "small text-muted", style: { wordBreak: "break-word" }, children: getDeviceShorthand(device) }) }),
                  /* @__PURE__ */ jsxRuntimeExports.jsx("td", { children: /* @__PURE__ */ jsxRuntimeExports.jsx(
                    "input",
                    {
                      type: "text",
                      className: "form-control form-control-sm",
                      placeholder: "Custom name",
                      value: nicknames[device.id] || "",
                      onChange: (e) => handleNicknameChange(device.id, e.target.value)
                    }
                  ) })
                ] }, device.id)) })
              ] }) })
            ] }) : /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "text-center text-muted py-4", children: "No devices available" }) }),
            /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "modal-footer", children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx(Button, { use: "secondary", onClick: onClose, children: "Cancel" }),
              /* @__PURE__ */ jsxRuntimeExports.jsx(
                Button,
                {
                  use: "primary",
                  onClick: handleSave,
                  disabled: !hasChanges || updateNickname.isPending,
                  children: /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { align: "center", gap: 1, children: [
                    updateNickname.isPending ? /* @__PURE__ */ jsxRuntimeExports.jsx(Spinner, { size: "sm" }) : /* @__PURE__ */ jsxRuntimeExports.jsx(Save, { size: 16 }),
                    /* @__PURE__ */ jsxRuntimeExports.jsx("span", { children: "Save Changes" })
                  ] })
                }
              )
            ] })
          ] })
        }
      )
    }
  );
};
class ErrorBoundary extends reactExports.Component {
  constructor(props) {
    super(props);
    __publicField(this, "handleReset", () => {
      this.setState({
        hasError: false,
        error: null,
        errorInfo: null
      });
    });
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null
    };
  }
  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }
  componentDidCatch(error, errorInfo) {
    console.error("ErrorBoundary caught an error:", error, errorInfo);
    this.setState({
      error,
      errorInfo
    });
  }
  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }
      return /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "container py-5", children: /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "card shadow", children: /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "card-body", children: /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", align: "center", gap: 3, className: "py-4", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx(AlertTriangle, { size: 64, className: "text-danger" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("h2", { className: "h4", children: "Something went wrong" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "text-muted text-center", children: "The application encountered an unexpected error. Don't worry, your data is safe." }),
        this.state.error && /* @__PURE__ */ jsxRuntimeExports.jsxs("details", { className: "w-100", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("summary", { className: "btn btn-sm btn-outline-secondary", children: "Show error details" }),
          /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "mt-3 p-3 bg-light border rounded", children: [
            /* @__PURE__ */ jsxRuntimeExports.jsxs("p", { className: "small mb-2", children: [
              /* @__PURE__ */ jsxRuntimeExports.jsx("strong", { children: "Error:" }),
              " ",
              this.state.error.toString()
            ] }),
            this.state.errorInfo && /* @__PURE__ */ jsxRuntimeExports.jsx("pre", { className: "small mb-0", style: { whiteSpace: "pre-wrap", wordBreak: "break-word" }, children: this.state.errorInfo.componentStack })
          ] })
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { gap: 2, children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx(Button, { use: "primary", onClick: this.handleReset, children: "Try Again" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx(Button, { use: "secondary", onClick: () => window.location.reload(), children: "Reload Page" })
        ] })
      ] }) }) }) });
    }
    return this.props.children;
  }
}
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false
    }
  }
});
function formatCaptureId(id2) {
  const match = id2.match(/^c(\d+)$/);
  return match ? `Capture ${match[1]}` : id2;
}
function CaptureTab({ capture, captureDevice: _captureDevice, isSelected, onClick, onDelete, onUpdateName, channelCount }) {
  const stateColor = capture.state === "running" ? "success" : capture.state === "failed" ? "danger" : "secondary";
  const [isEditing, setIsEditing] = reactExports.useState(false);
  const [editValue, setEditValue] = reactExports.useState("");
  const inputRef = reactExports.useRef(null);
  const displayName = capture.name || capture.autoName || formatCaptureId(capture.id);
  const hasCustomName = !!capture.name;
  reactExports.useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [isEditing]);
  const handleStartEdit = (e) => {
    e.stopPropagation();
    setEditValue(capture.name || capture.autoName || "");
    setIsEditing(true);
  };
  const handleSaveEdit = () => {
    const trimmedValue = editValue.trim();
    onUpdateName(trimmedValue || null);
    setIsEditing(false);
  };
  const handleKeyDown = (e) => {
    if (e.key === "Enter") {
      handleSaveEdit();
    } else if (e.key === "Escape") {
      setIsEditing(false);
    }
  };
  return /* @__PURE__ */ jsxRuntimeExports.jsxs(
    "button",
    {
      className: `btn btn-sm d-flex align-items-center gap-2 ${isSelected ? "btn-light" : "btn-outline-light"}`,
      onClick,
      style: {
        position: "relative",
        borderRadius: "0.375rem 0.375rem 0 0",
        borderBottom: "none",
        whiteSpace: "nowrap"
      },
      children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: `badge bg-${stateColor}`, style: { width: "8px", height: "8px", padding: 0, borderRadius: "50%" } }),
        isEditing ? /* @__PURE__ */ jsxRuntimeExports.jsx(
          "input",
          {
            ref: inputRef,
            type: "text",
            className: "form-control form-control-sm",
            style: { width: "120px", height: "20px", fontSize: "12px", padding: "2px 6px" },
            value: editValue,
            onChange: (e) => setEditValue(e.target.value),
            onBlur: handleSaveEdit,
            onKeyDown: handleKeyDown,
            onClick: (e) => e.stopPropagation()
          }
        ) : /* @__PURE__ */ jsxRuntimeExports.jsx(
          "span",
          {
            className: `fw-semibold ${isSelected ? "text-dark" : "text-white"}`,
            title: hasCustomName && capture.autoName ? `Auto: ${capture.autoName}` : void 0,
            children: displayName
          }
        ),
        !isEditing && /* @__PURE__ */ jsxRuntimeExports.jsx(
          "button",
          {
            className: `btn btn-sm p-0 ${isSelected ? "text-dark" : "text-white"}`,
            style: { width: "14px", height: "14px", lineHeight: 1 },
            onClick: handleStartEdit,
            title: "Edit name",
            children: /* @__PURE__ */ jsxRuntimeExports.jsx(Pen, { size: 10 })
          }
        ),
        /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: `small ${isSelected ? "text-muted" : "text-white opacity-75"}`, children: [
          formatFrequencyMHz(capture.centerHz),
          " MHz • ",
          channelCount,
          " ch"
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsx(
          "button",
          {
            className: `btn btn-sm p-0 ms-1 ${isSelected ? "text-dark" : "text-white"}`,
            style: { width: "16px", height: "16px", lineHeight: 1 },
            onClick: (e) => {
              e.stopPropagation();
              onDelete();
            },
            title: "Delete capture",
            children: /* @__PURE__ */ jsxRuntimeExports.jsx(X, { size: 12 })
          }
        )
      ]
    }
  );
}
function CaptureTabWithData({ capture, devices, isSelected, onClick, onDelete, onUpdateName }) {
  const { data: channels } = useChannels(capture.id);
  const captureDevice = devices == null ? void 0 : devices.find((d) => d.id === capture.deviceId);
  const channelCount = (channels == null ? void 0 : channels.length) ?? 0;
  return /* @__PURE__ */ jsxRuntimeExports.jsx(
    CaptureTab,
    {
      capture,
      captureDevice,
      isSelected,
      onClick,
      onDelete,
      onUpdateName,
      channelCount
    }
  );
}
function AppContent() {
  const { data: devices, isLoading: devicesLoading } = useDevices();
  const { data: captures, isLoading: capturesLoading } = useCaptures();
  const createCapture2 = useCreateCapture();
  const deleteCapture2 = useDeleteCapture();
  const updateCapture2 = useUpdateCapture();
  const toast = useToast();
  const [selectedCaptureId, setSelectedCaptureId] = reactExports.useState(() => {
    const params = new URLSearchParams(window.location.search);
    return params.get("capture");
  });
  const [showNewCaptureModal, setShowNewCaptureModal] = reactExports.useState(false);
  const [showWizard, setShowWizard] = reactExports.useState(false);
  const [showDeviceSettings, setShowDeviceSettings] = reactExports.useState(false);
  const [newCaptureDeviceId, setNewCaptureDeviceId] = reactExports.useState("");
  const [newCaptureFreq, setNewCaptureFreq] = reactExports.useState(1e8);
  reactExports.useEffect(() => {
    if (selectedCaptureId) {
      const params = new URLSearchParams(window.location.search);
      params.set("capture", selectedCaptureId);
      const newUrl = `${window.location.pathname}?${params.toString()}`;
      window.history.pushState({}, "", newUrl);
    } else if (captures == null ? void 0 : captures[0]) {
      setSelectedCaptureId(captures[0].id);
    }
  }, [selectedCaptureId, captures]);
  const selectedCapture = (captures == null ? void 0 : captures.find((c) => c.id === selectedCaptureId)) ?? (captures == null ? void 0 : captures[0]);
  const selectedDevice = devices == null ? void 0 : devices.find((d) => d.id === (selectedCapture == null ? void 0 : selectedCapture.deviceId));
  const { data: selectedCaptureChannels } = useChannels(selectedCapture == null ? void 0 : selectedCapture.id);
  const handleFrequencyClick = (frequencyHz) => {
    if (!selectedCapture)
      return;
    updateCapture2.mutate({
      captureId: selectedCapture.id,
      request: {
        centerHz: frequencyHz
      }
    }, {
      onSuccess: () => {
        toast.success(`Tuned to ${(frequencyHz / 1e6).toFixed(3)} MHz`);
      },
      onError: (error) => {
        toast.error((error == null ? void 0 : error.message) || "Failed to tune frequency");
      }
    });
  };
  const handleCreateCapture = () => {
    if (!newCaptureDeviceId)
      return;
    const device = devices == null ? void 0 : devices.find((d) => d.id === newCaptureDeviceId);
    const sampleRate = (device == null ? void 0 : device.sampleRates[0]) ?? 2e6;
    createCapture2.mutate({
      deviceId: newCaptureDeviceId,
      centerHz: newCaptureFreq,
      sampleRate
    }, {
      onSuccess: (newCapture) => {
        setShowNewCaptureModal(false);
        setSelectedCaptureId(newCapture.id);
        toast.success("Capture created successfully");
      },
      onError: (error) => {
        toast.error((error == null ? void 0 : error.message) || "Failed to create capture");
      }
    });
  };
  if (devicesLoading || capturesLoading) {
    return /* @__PURE__ */ jsxRuntimeExports.jsx(Flex, { justify: "center", align: "center", className: "min-vh-100", children: /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", align: "center", gap: 3, children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx(Spinner, {}),
      /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "text-muted", children: "Loading..." })
    ] }) });
  }
  if (devices && devices.length > 0 && !newCaptureDeviceId) {
    setNewCaptureDeviceId(devices[0].id);
  }
  return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "min-vh-100 bg-light", children: [
    /* @__PURE__ */ jsxRuntimeExports.jsx("nav", { className: "navbar navbar-dark bg-primary shadow-sm mb-0", style: { paddingBottom: 0 }, children: /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "container-fluid", style: { flexDirection: "column", alignItems: "stretch", gap: "0.5rem" }, children: [
      /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { align: "center", justify: "between", style: { paddingBottom: "0.5rem" }, children: [
        /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { align: "center", gap: 2, className: "text-white", children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx(Radio, { size: 24 }),
          /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "navbar-brand mb-0 h5 text-white", children: "WaveCap SDR" })
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { align: "center", gap: 2, children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            Button,
            {
              use: "light",
              size: "sm",
              onClick: () => setShowDeviceSettings(true),
              title: "Device Settings",
              children: /* @__PURE__ */ jsxRuntimeExports.jsx(Settings, { size: 16 })
            }
          ),
          /* @__PURE__ */ jsxRuntimeExports.jsxs("span", { className: "badge bg-light text-dark", children: [
            (devices == null ? void 0 : devices.length) ?? 0,
            " device",
            (devices == null ? void 0 : devices.length) !== 1 ? "s" : ""
          ] })
        ] })
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { align: "end", gap: 2, style: { marginBottom: "-1px" }, children: [
        captures && captures.length > 0 && /* @__PURE__ */ jsxRuntimeExports.jsx(jsxRuntimeExports.Fragment, { children: captures.map((capture) => /* @__PURE__ */ jsxRuntimeExports.jsx(
          CaptureTabWithData,
          {
            capture,
            devices,
            isSelected: (selectedCapture == null ? void 0 : selectedCapture.id) === capture.id,
            onClick: () => setSelectedCaptureId(capture.id),
            onDelete: () => {
              deleteCapture2.mutate(capture.id, {
                onSuccess: () => {
                  toast.success("Capture deleted successfully");
                },
                onError: (error) => {
                  toast.error((error == null ? void 0 : error.message) || "Failed to delete capture");
                }
              });
              if (selectedCaptureId === capture.id) {
                setSelectedCaptureId(null);
              }
            },
            onUpdateName: (name) => {
              updateCapture2.mutate({
                captureId: capture.id,
                request: { name }
              }, {
                onSuccess: () => {
                  toast.success("Capture name updated");
                },
                onError: (error) => {
                  toast.error((error == null ? void 0 : error.message) || "Failed to update capture name");
                }
              });
            }
          },
          capture.id
        )) }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { gap: 1, children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            Button,
            {
              use: "success",
              size: "sm",
              onClick: () => setShowWizard(true),
              title: "Use Recipe Wizard",
              children: /* @__PURE__ */ jsxRuntimeExports.jsx(Wand2, { size: 16 })
            }
          ),
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            Button,
            {
              use: "light",
              size: "sm",
              onClick: () => setShowNewCaptureModal(true),
              title: "Manual Setup",
              children: /* @__PURE__ */ jsxRuntimeExports.jsx(Plus, { size: 16 })
            }
          )
        ] })
      ] })
    ] }) }),
    /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "container-fluid px-4 py-3", style: { height: "calc(100vh - 120px)" }, children: selectedCapture ? /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "row g-2", style: { height: "100%" }, children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "col-12 col-lg-6 col-xxl-4", style: { height: "100%" }, children: /* @__PURE__ */ jsxRuntimeExports.jsx("div", { style: { height: "100%", overflowY: "auto", paddingRight: "4px" }, children: /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 2, children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx(ErrorBoundary, { children: /* @__PURE__ */ jsxRuntimeExports.jsx(
          SpectrumAnalyzer,
          {
            capture: selectedCapture,
            channels: selectedCaptureChannels,
            height: 200,
            onFrequencyClick: handleFrequencyClick
          }
        ) }),
        /* @__PURE__ */ jsxRuntimeExports.jsx(ErrorBoundary, { children: /* @__PURE__ */ jsxRuntimeExports.jsx(
          WaterfallDisplay,
          {
            capture: selectedCapture,
            channels: selectedCaptureChannels,
            height: 300,
            timeSpanSeconds: 30,
            colorScheme: "heat",
            intensity: 1.2
          }
        ) })
      ] }) }) }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "col-12 col-lg-6 col-xxl-4", style: { height: "100%" }, children: /* @__PURE__ */ jsxRuntimeExports.jsx("div", { style: { height: "100%", overflowY: "auto", paddingRight: "4px" }, children: /* @__PURE__ */ jsxRuntimeExports.jsx(ErrorBoundary, { children: /* @__PURE__ */ jsxRuntimeExports.jsx(RadioTuner, { capture: selectedCapture, device: selectedDevice }) }) }) }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "col-12 col-xxl-4", style: { height: "100%" }, children: /* @__PURE__ */ jsxRuntimeExports.jsx("div", { style: { height: "100%", overflowY: "auto", paddingRight: "4px" }, children: /* @__PURE__ */ jsxRuntimeExports.jsx(ErrorBoundary, { children: /* @__PURE__ */ jsxRuntimeExports.jsx(ChannelManager, { capture: selectedCapture }) }) }) })
    ] }) : /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "card shadow-sm", children: /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "card-body text-center py-5", children: /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "text-muted", children: "No captures available. Click + to create one or use the Wizard." }) }) }) }),
    showNewCaptureModal && /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "modal d-block", style: { backgroundColor: "rgba(0,0,0,0.5)" }, onClick: () => setShowNewCaptureModal(false), children: /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "modal-dialog modal-dialog-centered", onClick: (e) => e.stopPropagation(), children: /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "modal-content", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "modal-header", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx("h5", { className: "modal-title", children: "New Capture" }),
        /* @__PURE__ */ jsxRuntimeExports.jsx("button", { type: "button", className: "btn-close", onClick: () => setShowNewCaptureModal(false) })
      ] }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "modal-body", children: /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 3, children: [
        /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 1, children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("label", { className: "form-label", children: "Device" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "select",
            {
              className: "form-select",
              value: newCaptureDeviceId,
              onChange: (e) => setNewCaptureDeviceId(e.target.value),
              children: devices == null ? void 0 : devices.map((device) => /* @__PURE__ */ jsxRuntimeExports.jsxs("option", { value: device.id, children: [
                device.driver.toUpperCase(),
                " - ",
                device.label.substring(0, 60)
              ] }, device.id))
            }
          )
        ] }),
        /* @__PURE__ */ jsxRuntimeExports.jsxs(Flex, { direction: "column", gap: 1, children: [
          /* @__PURE__ */ jsxRuntimeExports.jsx("label", { className: "form-label", children: "Frequency (MHz)" }),
          /* @__PURE__ */ jsxRuntimeExports.jsx(
            "input",
            {
              type: "number",
              className: "form-control",
              value: (newCaptureFreq / 1e6).toFixed(3),
              onChange: (e) => setNewCaptureFreq(parseFloat(e.target.value) * 1e6),
              step: "0.1"
            }
          )
        ] })
      ] }) }),
      /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "modal-footer", children: [
        /* @__PURE__ */ jsxRuntimeExports.jsx(
          Button,
          {
            use: "secondary",
            onClick: () => setShowNewCaptureModal(false),
            children: "Cancel"
          }
        ),
        /* @__PURE__ */ jsxRuntimeExports.jsx(
          Button,
          {
            use: "success",
            onClick: handleCreateCapture,
            disabled: createCapture2.isPending,
            children: "Create"
          }
        )
      ] })
    ] }) }) }),
    showWizard && /* @__PURE__ */ jsxRuntimeExports.jsx(
      CreateCaptureWizard,
      {
        onClose: () => setShowWizard(false),
        onSuccess: (captureId) => {
          setSelectedCaptureId(captureId);
          setShowWizard(false);
        }
      }
    ),
    showDeviceSettings && /* @__PURE__ */ jsxRuntimeExports.jsx(
      DeviceSettingsModal,
      {
        onClose: () => setShowDeviceSettings(false)
      }
    )
  ] });
}
function App() {
  return /* @__PURE__ */ jsxRuntimeExports.jsx(QueryClientProvider, { client: queryClient, children: /* @__PURE__ */ jsxRuntimeExports.jsx(ToastProvider, { children: /* @__PURE__ */ jsxRuntimeExports.jsx(ErrorBoundary, { children: /* @__PURE__ */ jsxRuntimeExports.jsx(AppContent, {}) }) }) });
}
const index = "";
client.createRoot(document.getElementById("root")).render(
  /* @__PURE__ */ jsxRuntimeExports.jsx(React.StrictMode, { children: /* @__PURE__ */ jsxRuntimeExports.jsx(App, {}) })
);
//# sourceMappingURL=index-b22cc20f.js.map
